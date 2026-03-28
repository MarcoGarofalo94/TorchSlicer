"""
WorkerServicer — gRPC service implementation for TorchSlicer workers.

Shared by centralized and P2P topologies. Centralised workers register
with a coordinator; P2P workers wait for init() from the driver node.
"""

import io
import os
import socket
import struct
import threading
import traceback
from concurrent import futures

import torch
from torch import nn, optim

from ..config import RunConfig
from ..core.split_layer import SplitLayer

_UNSET = object()  # sentinel: distinguish "not provided" from explicit None
from ..monitor import tracer as _tracer
from ..monitor import WorkerProfiler
from ..monitor.process_logger import get_logger, configure as configure_process_logging
from ..retry import RetryPolicy

try:
    import grpc
    from torchslicer.transport.grpc.worker import worker_service_pb2, worker_service_pb2_grpc
    from torchslicer.transport.grpc.coordinator import (
        coordinator_service_pb2,
        coordinator_service_pb2_grpc,
    )
    _GRPC_AVAILABLE = True
except ImportError:
    _GRPC_AVAILABLE = False

_MAX_MSG = 256 * 1024 * 1024
_GRPC_OPTS = [
    ('grpc.max_send_message_length',    _MAX_MSG),
    ('grpc.max_receive_message_length', _MAX_MSG),
]

_LOG = get_logger("torchslicer.worker")
# TCP tensor frame format (big-endian):
#   kind(1c) | batch_id(4I) | dtype(1B) | ndim(2H) | ctx_len(2H)
#   followed by: ctx_bytes(ctx_len) | shape(ndim×8q) | payload_len(8Q) | payload
# ctx_len=0 when tracing is disabled — the two header bytes are the only overhead.
_FRAME_HEADER = struct.Struct("!cIBHH")
_SHAPE_ITEM = struct.Struct("!q")
_PAYLOAD_LEN = struct.Struct("!Q")
_HOSTNAME = socket.gethostname()


def _channel(addr):
    return grpc.insecure_channel(addr, options=_GRPC_OPTS)


class _StaleWorkError(RuntimeError):
    """Raised when queued/in-flight work belongs to a superseded worker config."""


class _PeerUnavailableError(RuntimeError):
    """Raised when a neighboring worker becomes unreachable mid-batch."""


class _TensorStreamError(RuntimeError):
    """Raised when the raw tensor transport fails."""


class _TensorStreamClosed(_TensorStreamError):
    """Raised when the raw tensor stream is closed cleanly by the peer."""


# ── dtype maps ─────────────────────────────────────────────────────────────────

_DTYPE_MAP: dict = {}
_TORCH_TO_DTYPE: dict = {}


def _init_dtype_maps():
    if not _GRPC_AVAILABLE:
        return
    mapping = [
        (worker_service_pb2.FLOAT32,  torch.float32),
        (worker_service_pb2.FLOAT16,  torch.float16),
        (worker_service_pb2.BFLOAT16, torch.bfloat16),
        (worker_service_pb2.FLOAT64,  torch.float64),
        (worker_service_pb2.INT64,    torch.int64),
        (worker_service_pb2.INT32,    torch.int32),
    ]
    for proto_t, torch_t in mapping:
        _DTYPE_MAP[proto_t]      = torch_t
        _TORCH_TO_DTYPE[torch_t] = proto_t


_init_dtype_maps()


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    arr = t.detach().cpu().contiguous()
    if arr.dtype == torch.bfloat16:
        return arr.view(torch.uint8).numpy().tobytes()
    return arr.numpy().tobytes()


def _tensor_payload_view(arr: torch.Tensor):
    if arr.dtype == torch.bfloat16:
        return memoryview(arr.view(torch.uint8).numpy()).cast("B")
    return memoryview(arr.numpy()).cast("B")


def deserialize_tensor(msg) -> torch.Tensor:
    dtype = _DTYPE_MAP.get(msg.dtype, torch.float32)
    return torch.frombuffer(bytearray(msg.data), dtype=dtype).reshape(list(msg.shape)).clone()


def serialize_tensor(t: torch.Tensor):
    with _tracer.span(
        "torchslicer.tensor.serialize",
        device=str(t.device),
        dtype=str(t.dtype),
        shape=str(tuple(t.shape)),
    ):
        data = _tensor_to_bytes(t)
    return worker_service_pb2.Tensor(
        data=data,
        shape=list(t.shape),
        dtype=_TORCH_TO_DTYPE.get(t.dtype, worker_service_pb2.FLOAT32),
    )


def _dtype_to_code(dtype: torch.dtype) -> int:
    return int(_TORCH_TO_DTYPE.get(dtype, worker_service_pb2.FLOAT32))


def _code_to_dtype(code: int) -> torch.dtype:
    return _DTYPE_MAP.get(code, torch.float32)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < n:
        data = sock.recv(n - len(chunks))
        if not data:
            if not chunks:
                raise _TensorStreamClosed("peer closed tensor stream")
            raise _TensorStreamError("unexpected EOF while reading tensor frame")
        chunks.extend(data)
    return bytes(chunks)


def _recv_into(sock: socket.socket, buf: bytearray) -> memoryview:
    view = memoryview(buf)
    offset = 0
    while offset < len(view):
        n_recv = sock.recv_into(view[offset:])
        if n_recv == 0:
            if offset == 0:
                raise _TensorStreamClosed("peer closed tensor stream")
            raise _TensorStreamError("unexpected EOF while reading tensor frame")
        offset += n_recv
    return view


def _load_transport_settings() -> tuple[str, int]:
    cfg = RunConfig.load(os.environ.get("EXPERIMENT_CONFIG"))
    tensor_transport = cfg.transport.tensor.strip().lower()
    if tensor_transport not in {"grpc", "tcp"}:
        raise ValueError(f"unsupported tensor transport: {tensor_transport!r}")
    return tensor_transport, int(cfg.transport.tensor_port_offset)


def _pack_tensor_frame(kind: bytes, batch_id: int, tensor: torch.Tensor):
    arr = tensor.detach().cpu().contiguous()
    payload = _tensor_payload_view(arr)
    shape = list(arr.shape)
    ctx_bytes = _tracer.inject_context_bytes()
    header = _FRAME_HEADER.pack(kind, int(batch_id), _dtype_to_code(arr.dtype), len(shape), len(ctx_bytes))
    shape_bytes = b"".join(_SHAPE_ITEM.pack(int(dim)) for dim in shape)
    payload_len = _PAYLOAD_LEN.pack(payload.nbytes)
    return header, ctx_bytes, shape_bytes, payload_len, payload


def _unpack_tensor_frame(sock: socket.socket) -> tuple[str, int, torch.Tensor, bytes]:
    header_buf = bytearray(_FRAME_HEADER.size)
    kind, batch_id, dtype_code, ndim, ctx_len = _FRAME_HEADER.unpack(_recv_into(sock, header_buf))
    ctx_bytes = bytes(_recv_into(sock, bytearray(ctx_len))) if ctx_len else b""
    shape_buf = bytearray(_SHAPE_ITEM.size * ndim)
    shape_view = _recv_into(sock, shape_buf)
    shape = [
        _SHAPE_ITEM.unpack_from(shape_view, i * _SHAPE_ITEM.size)[0]
        for i in range(ndim)
    ]
    payload_len_buf = bytearray(_PAYLOAD_LEN.size)
    payload_len = _PAYLOAD_LEN.unpack(_recv_into(sock, payload_len_buf))[0]
    payload = bytearray(payload_len)
    payload_view = _recv_into(sock, payload)
    dtype = _code_to_dtype(dtype_code)
    tensor = torch.frombuffer(payload_view, dtype=dtype).reshape(shape)
    return kind.decode("ascii"), batch_id, tensor, ctx_bytes


def _tensor_addr(addr: str, offset: int) -> str:
    host, port = addr.rsplit(":", 1)
    return f"{host}:{int(port) + offset}"


class _TensorPeerClient:
    def __init__(self, address: str):
        self.address = address
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def send_tensor(self, kind: bytes, batch_id: int, tensor: torch.Tensor) -> None:
        header, ctx_bytes, shape_bytes, payload_len, payload = _pack_tensor_frame(kind, batch_id, tensor)
        # Coalesce the small fixed-size fields into one syscall; payload is sent separately
        # because it can be hundreds of KB and is already a memoryview (zero-copy).
        meta = header + ctx_bytes + shape_bytes + payload_len
        with self._lock:
            self._ensure_connected()
            try:
                self._sock.sendall(meta)
                self._sock.sendall(payload)
            except Exception as exc:
                self.close()
                raise _TensorStreamError(f"tensor send failed peer={self.address}") from exc

    def _ensure_connected(self) -> None:
        if self._sock is not None:
            return
        host, port = self.address.rsplit(":", 1)
        sock = socket.create_connection((host, int(port)), timeout=10.0)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.close()
        finally:
            self._sock = None


def resolve_device() -> str:
    """Resolve the target device from the DEVICE environment variable.

    DEVICE=auto  (default) — use ``cuda`` if available, else ``mps`` if
                             available, else ``cpu``
    DEVICE=cuda            — prefer CUDA; falls back to ``cpu`` when CUDA is
                             unavailable
    DEVICE=mps             — prefer Apple Metal (MPS); falls back to ``cpu``
                             when MPS is unavailable
    DEVICE=cpu             — always use CPU regardless of hardware
    """
    import os
    val = os.environ.get("DEVICE", "auto").lower().strip()

    def _mps_available() -> bool:
        return (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        )

    def _auto_pick() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"

    if val == "auto":
        return _auto_pick()
    if val == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        _LOG.warning("DEVICE=cuda requested but CUDA is unavailable; falling back to cpu")
        return "cpu"
    if val == "mps":
        if _mps_available():
            return "mps"
        _LOG.warning("DEVICE=mps requested but MPS is unavailable; falling back to cpu")
        return "cpu"
    if val == "cpu":
        return "cpu"

    picked = _auto_pick()
    _LOG.warning("unknown DEVICE=%r; using %s", val, picked)
    return picked


def resolve_worker_address(port: int, hostname: str | None = None) -> str:
    """Resolve the worker address advertised during coordinator registration.

    Default is ``hostname:port`` from the current runtime environment.
    ``WORKER_ADDRESS`` can override this when the worker is reachable through a
    different host/IP than its local hostname (for example from inside a
    container or behind custom networking).
    """
    import os
    import socket

    host = hostname or socket.gethostname()
    advertised = os.environ.get("WORKER_ADDRESS", "").strip()
    if advertised:
        return advertised if ":" in advertised else f"{advertised}:{port}"
    return f"{host}:{port}"


def get_available_memory_mb(device: str) -> int:
    try:
        if device.startswith("cuda"):
            dev_idx = int(device.split(":")[-1]) if ":" in device else 0
            free, _ = torch.cuda.mem_get_info(dev_idx)
            return free // (1024 * 1024)
        import psutil
        return psutil.virtual_memory().available // (1024 * 1024)
    except Exception:
        return 0


def current_mem_mb(device) -> float:
    try:
        if device.type == "cuda":
            return torch.cuda.memory_allocated(device) / (1024 * 1024)
    except Exception:
        pass
    return 0.0


# ── WorkerServicer ─────────────────────────────────────────────────────────────

class WorkerServicer(
    worker_service_pb2_grpc.WorkerServiceServicer if _GRPC_AVAILABLE else object
):
    def __init__(self):
        super().__init__()
        self.device = torch.device(resolve_device())
        self._server = None
        self._generation = 0
        self._tensor_transport = "grpc"
        self._tensor_port_offset = 1
        self._tensor_server = None
        self._tensor_stop = threading.Event()
        self._tensor_threads: list[threading.Thread] = []

        self._reset_state()

        # Single-worker compute pool: serialises forward/backward on this worker
        # so concurrent micro-batch RPCs never race on self.layer.x or .grad tensors.
        self._pool = futures.ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()

    def set_server(self, server):
        self._server = server

    def start_tensor_server(self, port: int):
        if self._tensor_transport != "tcp":
            return
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", int(port)))
        server.listen()
        self._tensor_server = server
        thread = threading.Thread(
            target=self._accept_tensor_loop,
            daemon=True,
            name=f"tensor-accept-{port}",
        )
        thread.start()
        self._tensor_threads.append(thread)
        _LOG.info("tensor transport listener started port=%s mode=tcp", port)

    def stop_tensor_server(self):
        self._tensor_stop.set()
        if self._tensor_server is not None:
            try:
                self._tensor_server.close()
            except Exception:
                pass
            self._tensor_server = None
        for peer in (self._next_tensor, self._prev_tensor):
            if peer is not None:
                peer.close()

    def _accept_tensor_loop(self):
        while not self._tensor_stop.is_set():
            try:
                conn, addr = self._tensor_server.accept()
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(1.0)  # allows _tensor_stop to be checked on shutdown
            thread = threading.Thread(
                target=self._handle_tensor_conn,
                args=(conn, addr),
                daemon=True,
            )
            thread.start()
            # Per-connection threads are daemon threads; do not track to avoid
            # unbounded list growth across reconnections.

    def _handle_tensor_conn(self, conn: socket.socket, addr):
        try:
            while not self._tensor_stop.is_set():
                try:
                    kind, batch_id, tensor, ctx_bytes = _unpack_tensor_frame(conn)
                except socket.timeout:
                    continue  # check _tensor_stop, then retry read
                generation = self._generation
                # Attach the sender's trace context so worker spans become
                # children of the coordinator's batch span (or the previous
                # worker's forward/backward span).  No-op when tracing is off.
                with _tracer.extract_context(ctx_bytes):
                    if kind == "F":
                        self._pool.submit(
                            _tracer.propagate_to_thread(self._forward_tensor),
                            batch_id, tensor, generation,
                        )
                    elif kind == "B":
                        self._pool.submit(
                            _tracer.propagate_to_thread(self._backward_tensor),
                            batch_id, tensor, generation,
                        )
                    else:
                        raise _TensorStreamError(f"unknown tensor frame kind={kind!r}")
        except Exception as exc:
            if not self._tensor_stop.is_set() and not isinstance(exc, _TensorStreamClosed):
                _LOG.warning("tensor transport connection closed addr=%s error=%s", addr, exc)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _reset_state(self):
        self.layer:       SplitLayer = None
        self.loss_fn                 = None
        self.is_last:     bool       = False
        self.prev_worker: str        = None
        self.next_worker: str        = None
        self.coordinator: str        = None
        self._n_micro:    int        = 1
        self._run_id:     str        = ""
        self._worker_index: int      = 0

        self._next_stub = None
        self._prev_stub = None
        self._next_tensor = None
        self._prev_tensor = None
        self._coord_stub = None
        self._fatal_error = None

        self._labels:       dict = {}
        self._outputs:      dict = {}
        self._micro_losses: dict = {}

        self._profiler: WorkerProfiler = WorkerProfiler(verbosity=0)

    # ── init ───────────────────────────────────────────────────────────────────

    def init(self, request, context):
        try:
            with self._lock:
                self._generation += 1
                self._reset_state()

            layers = self._build_layers(request.layers)
            predecessors = (
                [list(p.indices) for p in request.predecessors]
                if request.predecessors else None
            )
            self.layer   = SplitLayer(layers, is_last=request.is_last, predecessors=predecessors)
            self.is_last = request.is_last
            self._run_id       = request.run_id
            self._worker_index = request.worker_index

            if request.is_last and request.HasField("criterion"):
                crit_extra = (
                    torch.load(io.BytesIO(request.criterion.extra_params), weights_only=False)
                    if request.criterion.extra_params else {}
                )
                self.loss_fn = getattr(nn, request.criterion.name)(**crit_extra)

            opt_extra = (
                torch.load(io.BytesIO(request.optimizer.extra_params), weights_only=False)
                if request.optimizer.extra_params else {}
            )
            trainable = [p for p in self.layer.parameters() if p.requires_grad]
            params = trainable if trainable else list(self.layer.parameters())
            opt = getattr(optim, request.optimizer.name)(
                params, lr=request.optimizer.lr, **opt_extra)
            self.layer.set_optimizer(opt)

            self.prev_worker = request.prev_worker or None
            self.next_worker = request.next_worker or None
            self.coordinator = request.coordinator
            self._n_micro    = max(1, request.n_micro) if request.n_micro else 1

            if self.next_worker:
                if self._tensor_transport == "tcp":
                    self._next_tensor = _TensorPeerClient(
                        _tensor_addr(self.next_worker, self._tensor_port_offset)
                    )
                else:
                    self._next_stub = worker_service_pb2_grpc.WorkerServiceStub(
                        _channel(self.next_worker))
            if self.prev_worker:
                if self._tensor_transport == "tcp":
                    self._prev_tensor = _TensorPeerClient(
                        _tensor_addr(self.prev_worker, self._tensor_port_offset)
                    )
                else:
                    self._prev_stub = worker_service_pb2_grpc.WorkerServiceStub(
                        _channel(self.prev_worker))
            if self.coordinator:
                self._coord_stub = coordinator_service_pb2_grpc.CoordinatorServiceStub(
                    _channel(self.coordinator))

            if request.checkpoint_path:
                self._load_checkpoint(request.checkpoint_path)

            self.layer = self.layer.to(self.device)
            self.layer.train()
            if self.loss_fn:
                self.loss_fn = self.loss_fn.to(self.device)

            self._profiler = WorkerProfiler(
                verbosity = int(request.profile_verbosity),
                memory    = bool(request.profile_memory),
                device    = self.device,
            )

            layer_names   = [type(l).__name__ for l in self.layer.layers]
            param_bytes   = sum(p.numel() * p.element_size() for p in self.layer.parameters())
            param_mb      = round(param_bytes / (1024 * 1024), 2)
            cuda_alloc_mb = round(
                torch.cuda.memory_allocated(self.device) / (1024 * 1024), 2
            ) if self.device.type == 'cuda' else 0.0

            _LOG.info(
                "worker init run_id=%s index=%s layers=%s is_last=%s device=%s prev=%s next=%s transport=%s param_mb=%.2f cuda_alloc_mb=%.2f n_micro=%s profile_verbosity=%s profile_memory=%s",
                self._run_id,
                self._worker_index,
                layer_names,
                self.is_last,
                self.device,
                self.prev_worker,
                self.next_worker,
                self._tensor_transport,
                param_mb,
                cuda_alloc_mb,
                self._n_micro,
                request.profile_verbosity,
                request.profile_memory,
            )

            with _tracer.span(
                "torchslicer.worker.init",
                worker=_HOSTNAME,
                layers=", ".join(layer_names),
                n_layers=len(layer_names),
                is_last=self.is_last,
                prev_worker=self.prev_worker or "",
                next_worker=self.next_worker or "",
                param_mb=param_mb,
                cuda_alloc_mb=cuda_alloc_mb,
            ):
                pass

            return worker_service_pb2.StatusMessage(
                ok=True, message="Initialized", hostname=_HOSTNAME)
        except Exception as e:
            _LOG.exception("worker init failed: %s", e)
            return worker_service_pb2.StatusMessage(
                ok=False, message=str(e), hostname=_HOSTNAME)

    # ── shutdown ───────────────────────────────────────────────────────────────

    def shutdown(self, request, context):
        def _do_shutdown():
            if request.save_checkpoint:
                try:
                    self._save_checkpoint(
                        checkpoint_dir=request.checkpoint_dir,
                        run_id=request.run_id,
                        epoch=request.epoch,
                        worker_index=request.worker_index,
                    )
                except Exception as e:
                    _LOG.exception("shutdown checkpoint save failed: %s", e)

            _LOG.info("worker stopping hostname=%s", _HOSTNAME)
            self.stop_tensor_server()
            if self._server:
                self._server.stop(grace=1)

        threading.Thread(target=_do_shutdown, daemon=True).start()
        return worker_service_pb2.Ack(batch_id=0)

    # ── save_checkpoint ────────────────────────────────────────────────────────

    def save_checkpoint(self, request, context):
        """Save slice checkpoint without stopping the server (fault-tolerance hook)."""
        if self.layer is not None:
            try:
                self._save_checkpoint(
                    checkpoint_dir=request.checkpoint_dir,
                    run_id=request.run_id,
                    epoch=request.epoch,
                    worker_index=request.worker_index,
                )
            except Exception as e:
                _LOG.warning("save_checkpoint failed: %s", e)
        return worker_service_pb2.Ack(batch_id=0)

    # ── get_stats ──────────────────────────────────────────────────────────────

    def get_stats(self, request, context):
        epoch   = request.epoch
        summary = self._profiler.epoch_summary(epoch)
        records = self._profiler.batch_records()
        self._profiler.reset_epoch()

        def _phase_stats(name: str):
            return worker_service_pb2.PhaseStats(
                avg_ms      = summary.get(f"{name}_avg_ms",   0.0),
                min_ms      = summary.get(f"{name}_min_ms",   0.0),
                max_ms      = summary.get(f"{name}_max_ms",   0.0),
                p95_ms      = summary.get(f"{name}_p95_ms",   0.0),
                total_ms    = summary.get(f"{name}_total_ms", 0.0),
                count       = summary.get("n_batches",        0),
                peak_mem_mb = summary.get(f"{name}_peak_mem_mb", 0.0),
            )

        batch_stats = [
            worker_service_pb2.BatchStats(
                batch_id     = r["batch_id"],
                forward_ms   = r["forward_ms"],
                backward_ms  = r["backward_ms"],
                optimizer_ms = r["optimizer_ms"],
                send_fwd_ms  = r["send_fwd_ms"],
                send_bwd_ms  = r["send_bwd_ms"],
                idle_fwd_ms  = r["idle_fwd_ms"],
                idle_bwd_ms  = r["idle_bwd_ms"],
                peak_mem_mb  = r["peak_mem_mb"],
            )
            for r in records
        ]

        return worker_service_pb2.WorkerStatsResponse(
            worker_index = self._worker_index,
            run_id       = self._run_id,
            epoch        = epoch,
            forward      = _phase_stats("forward"),
            backward     = _phase_stats("backward"),
            optimizer    = _phase_stats("optimizer"),
            send_fwd     = _phase_stats("send_fwd"),
            send_bwd     = _phase_stats("send_bwd"),
            idle_fwd     = _phase_stats("idle_fwd"),
            idle_bwd     = _phase_stats("idle_bwd"),
            peak_mem_mb  = summary.get("forward_peak_mem_mb", 0.0),
            end_mem_mb   = round(current_mem_mb(self.device), 2),
            n_batches    = summary.get("n_batches", 0),
            batches      = batch_stats,
        )

    # ── forward / backward RPCs (fire-and-forget) ──────────────────────────────

    def forward(self, request, context):
        generation = self._generation
        self._pool.submit(self._forward, request, generation)
        return worker_service_pb2.Ack(batch_id=request.batch_id)

    def backward(self, request, context):
        generation = self._generation
        self._pool.submit(self._backward, request, generation)
        return worker_service_pb2.Ack(batch_id=request.batch_id)

    # ── internal forward ───────────────────────────────────────────────────────

    def _before_runtime_phase(self, phase: str, batch_id: int) -> None:
        """Test hook for injecting runtime faults in subclasses."""

    def _ensure_generation(self, generation: int, phase: str, batch_id: int) -> None:
        if generation != self._generation:
            raise _StaleWorkError(
                f"stale work phase={phase} batch_id={batch_id} "
                f"generation={generation} current={self._generation}"
            )

    @staticmethod
    def _is_unavailable_rpc(exc: Exception) -> bool:
        return (
            _GRPC_AVAILABLE
            and isinstance(exc, grpc.RpcError)
            and exc.code() == grpc.StatusCode.UNAVAILABLE
        )

    def _forward(self, request, generation: int):
        try:
            batch_id = request.batch_id
            self._ensure_generation(generation, "forward", batch_id)
            self._before_runtime_phase("forward", batch_id)
            self._profiler.begin_batch(batch_id)
            self._profiler.mark_idle_end("fwd")

            if self.is_last:
                self._forward_last(batch_id, request, generation)
                return

            with _tracer.span(
                "torchslicer.tensor.deserialize",
                batch_id=batch_id,
                worker=_HOSTNAME,
                source="input",
            ):
                tensor_cpu = deserialize_tensor(request.input)
            with _tracer.span(
                "torchslicer.tensor.h2d",
                batch_id=batch_id,
                worker=_HOSTNAME,
                source="input",
                device=str(self.device),
            ):
                tensor = tensor_cpu.to(self.device)
            # Unpack aux inputs (multimodal models — sent to first worker only)
            aux = {}
            if request.aux_inputs:
                for k, v in request.aux_inputs.items():
                    with _tracer.span(
                        "torchslicer.tensor.deserialize",
                        batch_id=batch_id,
                        worker=_HOSTNAME,
                        source=f"aux:{k}",
                    ):
                        aux_cpu = deserialize_tensor(v)
                    with _tracer.span(
                        "torchslicer.tensor.h2d",
                        batch_id=batch_id,
                        worker=_HOSTNAME,
                        source=f"aux:{k}",
                        device=str(self.device),
                    ):
                        aux[k] = aux_cpu.to(self.device)
            self._run_forward_stage(batch_id, tensor, generation, aux=aux)

        except _StaleWorkError as e:
            _LOG.info("%s", e)
        except _PeerUnavailableError as e:
            _LOG.warning("%s", e)
        except Exception as e:
            self._report_fatal_error("forward", request.batch_id, e)

    def _forward_tensor(self, batch_id: int, tensor_cpu: torch.Tensor, generation: int):
        try:
            self._ensure_generation(generation, "forward_tensor", batch_id)
            self._before_runtime_phase("forward_tensor", batch_id)
            if not self._profiler.batch_active(batch_id):
                self._profiler.begin_batch(batch_id)
            self._profiler.mark_idle_end("fwd")
            with _tracer.span(
                "torchslicer.tensor.h2d",
                batch_id=batch_id,
                worker=_HOSTNAME,
                source="input_tcp",
                device=str(self.device),
            ):
                tensor = tensor_cpu.to(self.device)
            self._run_forward_stage(batch_id, tensor, generation, aux={})
        except _StaleWorkError as e:
            _LOG.info("%s", e)
        except _PeerUnavailableError as e:
            _LOG.warning("%s", e)
        except Exception as e:
            self._report_fatal_error("forward_tcp", batch_id, e)

    def _run_forward_stage(self, batch_id: int, tensor: torch.Tensor, generation: int, aux: dict):
        with _tracer.span(
            "torchslicer.worker.forward",
            kind="TOOL",
            batch_id=batch_id,
            worker=_HOSTNAME,
            is_last=self.is_last,
            input_shape=str(tuple(tensor.shape)),
        ) as s:
            with self._profiler.phase("forward"):
                out = self.layer(tensor, **aux) if aux else self.layer(tensor)
                x_ref = self.layer.x
            if s:
                s.set_attribute("output_shape", str(tuple(out.shape)))

        with self._lock:
            self._ensure_generation(generation, "forward", batch_id)
            if self.is_last:
                label = self._labels.pop(batch_id, None)
                if label is None:
                    self._outputs[batch_id] = (out, x_ref)
            else:
                self._outputs[batch_id] = (out, x_ref)

        if self.is_last:
            if label is not None:
                self._run_backward_last(batch_id, out, x_ref, label, generation)
            return

        self._ensure_generation(generation, "forward", batch_id)
        with self._profiler.phase("send_fwd"):
            self._send_forward_peer(batch_id, out)

        self._profiler.mark_idle_start("bwd")

    def _send_forward_peer(self, batch_id: int, out: torch.Tensor) -> None:
        try:
            if self._tensor_transport == "tcp" and self._next_tensor is not None:
                with _tracer.span(
                    "torchslicer.tcp.forward_send",
                    batch_id=batch_id,
                    worker=_HOSTNAME,
                    peer=self.next_worker,
                ):
                    self._next_tensor.send_tensor(b"F", batch_id, out)
                return
            with _tracer.span(
                "torchslicer.rpc.forward_send",
                batch_id=batch_id,
                worker=_HOSTNAME,
                peer=self.next_worker,
            ):
                self._next_stub.forward(worker_service_pb2.ForwardRequest(
                    batch_id=batch_id,
                    input=serialize_tensor(out),
                ))
        except Exception as exc:
            if isinstance(exc, _TensorStreamError) or self._is_unavailable_rpc(exc):
                raise _PeerUnavailableError(
                    f"peer unavailable phase=forward batch_id={batch_id} peer={self.next_worker}"
                ) from exc
            raise

    def _forward_last(self, batch_id: int, request, generation: int):
        try:
            self._ensure_generation(generation, "forward_last", batch_id)
            self._before_runtime_phase("forward_last", batch_id)
            is_label = bool(request.label.data)

            if is_label:
                with _tracer.span(
                    "torchslicer.tensor.deserialize",
                    batch_id=batch_id,
                    worker=_HOSTNAME,
                    source="label",
                ):
                    label_cpu = deserialize_tensor(request.label)
                with _tracer.span(
                    "torchslicer.tensor.h2d",
                    batch_id=batch_id,
                    worker=_HOSTNAME,
                    source="label",
                    device=str(self.device),
                ):
                    label = label_cpu.to(self.device)
                with self._lock:
                    self._ensure_generation(generation, "forward_last", batch_id)
                    self._labels[batch_id] = label
                    cached = self._outputs.pop(batch_id, None)
                if cached is not None:
                    out, x_ref = cached
                    self._run_backward_last(batch_id, out, x_ref, label, generation)
            else:
                with _tracer.span(
                    "torchslicer.tensor.deserialize",
                    batch_id=batch_id,
                    worker=_HOSTNAME,
                    source="input_last",
                ):
                    tensor_cpu = deserialize_tensor(request.input)
                with _tracer.span(
                    "torchslicer.tensor.h2d",
                    batch_id=batch_id,
                    worker=_HOSTNAME,
                    source="input_last",
                    device=str(self.device),
                ):
                    tensor = tensor_cpu.to(self.device)
                with _tracer.span(
                    "torchslicer.worker.forward",
                    batch_id=batch_id,
                    worker=_HOSTNAME,
                    is_last=True,
                    input_shape=str(tuple(tensor.shape)),
                ):
                    with self._profiler.phase("forward"):
                        out   = self.layer(tensor)
                        x_ref = self.layer.x

                with self._lock:
                    self._ensure_generation(generation, "forward_last", batch_id)
                    label = self._labels.pop(batch_id, None)
                    if label is None:
                        self._outputs[batch_id] = (out, x_ref)

                if label is not None:
                    self._run_backward_last(batch_id, out, x_ref, label, generation)
        except _StaleWorkError as e:
            _LOG.info("%s", e)
        except Exception as e:
            self._report_fatal_error("forward_last", batch_id, e)

    # ── internal backward ──────────────────────────────────────────────────────

    def _backward(self, request, generation: int):
        try:
            batch_id      = request.batch_id
            self._ensure_generation(generation, "backward", batch_id)
            self._before_runtime_phase("backward", batch_id)
            n_micro       = self._n_micro
            is_last_micro = (n_micro <= 1) or (batch_id % n_micro == n_micro - 1)

            self._profiler.mark_idle_end("bwd")

            with _tracer.span(
                "torchslicer.tensor.deserialize",
                batch_id=batch_id,
                worker=_HOSTNAME,
                source="gradient",
            ):
                grad_cpu = deserialize_tensor(request.gradient)
            with _tracer.span(
                "torchslicer.tensor.h2d",
                batch_id=batch_id,
                worker=_HOSTNAME,
                source="gradient",
                device=str(self.device),
            ):
                grad_in = grad_cpu.to(self.device)
            self._run_backward_stage(batch_id, grad_in, is_last_micro, generation)
        except _StaleWorkError as e:
            _LOG.info("%s", e)
        except _PeerUnavailableError as e:
            _LOG.warning("%s", e)
        except Exception as e:
            self._report_fatal_error("backward", request.batch_id, e)

    def _backward_tensor(self, batch_id: int, grad_cpu: torch.Tensor, generation: int):
        try:
            self._ensure_generation(generation, "backward_tensor", batch_id)
            self._before_runtime_phase("backward_tensor", batch_id)
            n_micro       = self._n_micro
            is_last_micro = (n_micro <= 1) or (batch_id % n_micro == n_micro - 1)
            self._profiler.mark_idle_end("bwd")
            with _tracer.span(
                "torchslicer.tensor.h2d",
                batch_id=batch_id,
                worker=_HOSTNAME,
                source="gradient_tcp",
                device=str(self.device),
            ):
                grad_in = grad_cpu.to(self.device)
            self._run_backward_stage(batch_id, grad_in, is_last_micro, generation)
        except _StaleWorkError as e:
            _LOG.info("%s", e)
        except _PeerUnavailableError as e:
            _LOG.warning("%s", e)
        except Exception as e:
            self._report_fatal_error("backward_tcp", batch_id, e)

    def _run_backward_stage(self, batch_id: int, grad_in: torch.Tensor, is_last_micro: bool, generation: int):
        with self._lock:
            self._ensure_generation(generation, "backward", batch_id)
            cached = self._outputs.pop(batch_id, None)

        if cached is None:
            _LOG.warning("backward skipped because cached output is missing for batch_id=%s", batch_id)
            return

        out, x_ref = cached

        with _tracer.span(
            "torchslicer.worker.backward",
            kind="TOOL",
            batch_id=batch_id,
            worker=_HOSTNAME,
            is_last=False,
        ):
            with self._profiler.phase("backward"):
                moe = self.layer.pop_moe_aux_loss() if is_last_micro else None
                if moe is not None:
                    out.backward(grad_in, retain_graph=True)
                    moe.backward()
                else:
                    out.backward(grad_in)
                grad = x_ref.grad if x_ref is not None else None
            if is_last_micro:
                with self._profiler.phase("optimizer"):
                    self.layer.optimize()

        del out
        self._send_backward(batch_id, grad, is_last_micro, generation)

    def _run_backward_last(self, batch_id: int, out: torch.Tensor,
                           x_ref: torch.Tensor, label: torch.Tensor, generation: int):
        try:
            self._ensure_generation(generation, "backward_last", batch_id)
            self._before_runtime_phase("backward_last", batch_id)
            n_micro       = self._n_micro
            is_last_micro = (n_micro <= 1) or (batch_id % n_micro == n_micro - 1)
            full_batch_id = batch_id // n_micro if n_micro > 1 else batch_id

            loss_unscaled = self.loss_fn(out, label)
            loss          = loss_unscaled / n_micro

            with self._lock:
                self._micro_losses[full_batch_id] = (
                    self._micro_losses.get(full_batch_id, 0.0) + loss_unscaled.item()
                )

            if is_last_micro:
                avg_loss = self._micro_losses.pop(full_batch_id) / n_micro
                self._ensure_generation(generation, "backward_last", batch_id)
                _stub, _run_id = self._coord_stub, self._run_id
                _msg = coordinator_service_pb2.MetricsMessage(
                    batch_id=batch_id,
                    loss=avg_loss,
                    worker=_HOSTNAME,
                    run_id=_run_id,
                )
                threading.Thread(
                    target=_tracer.propagate_to_thread(lambda m=_msg: _stub.report_metrics(m)),
                    daemon=True,
                ).start()

            with _tracer.span(
                "torchslicer.worker.backward",
                batch_id=batch_id,
                worker=_HOSTNAME,
                is_last=True,
                loss=loss_unscaled.item(),
            ):
                with self._profiler.phase("backward"):
                    # Pop MoE aux loss BEFORE backward — add to loss so both
                    # backward through the same graph in one pass.
                    moe = self.layer.pop_moe_aux_loss() if is_last_micro else None
                    total_loss = loss + moe if moe is not None else loss
                    total_loss.backward()
                    grad = x_ref.grad
                if is_last_micro:
                    with self._profiler.phase("optimizer"):
                        self.layer.optimize()

            del out, label, loss, loss_unscaled
            self._send_backward(batch_id, grad, is_last_micro, generation)
        except _StaleWorkError as e:
            _LOG.info("%s", e)
        except _PeerUnavailableError as e:
            _LOG.warning("%s", e)
        except Exception as e:
            self._report_fatal_error("backward_last", batch_id, e)

    def _send_backward(self, batch_id: int, grad: torch.Tensor, is_last_micro: bool = True,
                       generation: int | None = None):
        if generation is not None:
            self._ensure_generation(generation, "send_backward", batch_id)
        if self._prev_stub or self._prev_tensor:
            with self._profiler.phase("send_bwd"):
                try:
                    if self._tensor_transport == "tcp" and self._prev_tensor is not None:
                        with _tracer.span(
                            "torchslicer.tcp.backward_send",
                            batch_id=batch_id,
                            worker=_HOSTNAME,
                            peer=self.prev_worker,
                        ):
                            self._prev_tensor.send_tensor(b"B", batch_id, grad)
                    else:
                        with _tracer.span(
                            "torchslicer.rpc.backward_send",
                            batch_id=batch_id,
                            worker=_HOSTNAME,
                            peer=self.prev_worker,
                        ):
                            self._prev_stub.backward(worker_service_pb2.BackwardRequest(
                                batch_id=batch_id,
                                gradient=serialize_tensor(grad),
                            ))
                except Exception as exc:
                    if isinstance(exc, _TensorStreamError) or self._is_unavailable_rpc(exc):
                        raise _PeerUnavailableError(
                            f"peer unavailable phase=backward batch_id={batch_id} peer={self.prev_worker}"
                        ) from exc
                    raise
            self._profiler.mark_idle_start("fwd")
            self._profiler.end_batch()
        elif is_last_micro:
            self._coord_stub.batch_done(coordinator_service_pb2.BatchDoneRequest(
                batch_id=batch_id,
                run_id=self._run_id,
            ))
            self._profiler.mark_idle_start("fwd")
            self._profiler.end_batch()

    def _report_fatal_error(self, phase: str, batch_id: int, exc: Exception):
        tb = traceback.format_exc()
        with self._lock:
            if self._fatal_error is not None:
                return
            self._fatal_error = {
                "phase": phase,
                "batch_id": batch_id,
                "message": str(exc),
            }

        _LOG.exception("fatal worker error phase=%s batch_id=%s error=%s", phase, batch_id, exc)

        if self._coord_stub and self._run_id:
            try:
                self._coord_stub.report_worker_error(
                    coordinator_service_pb2.WorkerError(
                        worker_index=self._worker_index,
                        batch_id=batch_id,
                        run_id=self._run_id,
                        worker=_HOSTNAME,
                        phase=phase,
                        message=str(exc),
                        traceback=tb,
                        fatal=True,
                    ),
                    timeout=5.0,
                )
            except Exception as report_exc:
                _LOG.warning("worker error reporting failed phase=%s batch_id=%s error=%s", phase, batch_id, report_exc)

        if self._server:
            threading.Thread(
                target=lambda: self._server.stop(grace=0),
                daemon=True,
                name="worker-fatal-stop",
            ).start()

    # ── checkpoint ─────────────────────────────────────────────────────────────

    def _save_checkpoint(self, checkpoint_dir: str, run_id: str, epoch: int, worker_index: int):
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"worker_{worker_index}_epoch_{epoch}.pt")
        torch.save({
            "layer_state_dict":     self.layer.state_dict(),
            "optimizer_state_dict": self.layer.optimizer.state_dict() if self.layer.optimizer else None,
            "epoch":                epoch,
            "worker_index":         worker_index,
            "run_id":               run_id,
        }, path)
        _LOG.info("checkpoint saved path=%s", path)

    def _load_checkpoint(self, checkpoint_path: str):
        import os
        if not os.path.exists(checkpoint_path):
            _LOG.warning("resume checkpoint not found path=%s; starting fresh", checkpoint_path)
            return
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.layer.load_state_dict(state["layer_state_dict"])
        if state.get("optimizer_state_dict") and self.layer.optimizer:
            self.layer.optimizer.load_state_dict(state["optimizer_state_dict"])
        _LOG.info("checkpoint loaded path=%s epoch=%s", checkpoint_path, state.get("epoch", "?"))

    def _build_layers(self, layer_configs) -> list:
        return [
            torch.load(io.BytesIO(lc.serialized), weights_only=False)
            for lc in layer_configs
        ]


# ── run_worker ─────────────────────────────────────────────────────────────────

def run_worker(
    port=None,
    coordinator_addr=_UNSET,
    worker_address=None,
    device=None,
    servicer_class=None,
    grpc_workers=10,
):
    """Start a TorchSlicer worker and block until shutdown.

    All parameters fall back to environment variables when not provided:

    Args:
        port:             Listening port. Env: ``PORT`` (default 50051).
        coordinator_addr: Coordinator gRPC address for worker registration.
                          Defaults to ``RunConfig.network.coordinator_address``
                          (env: ``COORDINATOR_ADDRESS``). Pass ``None`` explicitly
                          to skip registration — required for P2P followers that
                          wait for ``init()`` from the driver instead of
                          self-registering.
        worker_address:   Address advertised to the coordinator / peers.
                          Env: ``WORKER_ADDRESS`` (default ``hostname:port``).
        device:           ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.
                          Env: ``DEVICE`` (default ``"auto"``).
                          When provided, overrides the env var for this call.
        servicer_class:   ``WorkerServicer`` subclass to instantiate.
                          Defaults to ``WorkerServicer``. Use this to inject
                          custom behaviour (e.g. fault-injection test workers)
                          without duplicating the server-startup boilerplate.
        grpc_workers:     gRPC thread-pool size (default 10).

    Example — centralized worker (3-liner)::

        import torchslicer as ts
        ts.run_worker()   # reads PORT / COORDINATOR_ADDRESS / DEVICE from env

    Example — P2P follower (no coordinator)::

        ts.run_worker(coordinator_addr=None)

    Example — custom servicer::

        class MyServicer(ts.WorkerServicer):
            ...

        ts.run_worker(servicer_class=MyServicer)
    """
    import os
    import socket
    import sys

    from ..monitor import tracer as _tracer

    _tracer.auto_configure_if_env()

    # ── load config (YAML + env overrides) ─────────────────────────────────
    cfg = RunConfig.load()
    configure_process_logging(cfg.logging.level)

    # ── resolve port ────────────────────────────────────────────────────────
    if port is None:
        port = cfg.network.worker_port
    port = int(port)

    # ── resolve coordinator address ─────────────────────────────────────────
    # Explicit coordinator_addr=None → always skip registration (P2P followers).
    # Default (_UNSET) → use config value; None config value → skip registration.
    if coordinator_addr is _UNSET:
        coordinator_addr = cfg.network.coordinator_address or None

    # ── resolve worker address ──────────────────────────────────────────────
    if worker_address is None:
        _raw = cfg.network.worker_address
        if _raw:
            worker_address = _raw if ":" in _raw else f"{_raw}:{port}"
        else:
            worker_address = resolve_worker_address(port, hostname=_HOSTNAME)

    # ── resolve device ──────────────────────────────────────────────────────
    if device is not None:
        os.environ["DEVICE"] = device
    elif cfg.network.device and cfg.network.device != "auto":
        os.environ["DEVICE"] = cfg.network.device
    resolved_device = resolve_device()
    memory_mb       = get_available_memory_mb(resolved_device)

    # ── resolve worker tags ─────────────────────────────────────────────────
    tags = list(cfg.network.worker_tags)

    # ── build servicer + gRPC server ────────────────────────────────────────
    tensor_transport    = cfg.transport.tensor.strip().lower()
    tensor_port_offset  = int(cfg.transport.tensor_port_offset)

    if servicer_class is None:
        servicer_class = WorkerServicer
    servicer = servicer_class()
    servicer._tensor_transport = tensor_transport
    servicer._tensor_port_offset = tensor_port_offset

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=grpc_workers),
        options=_GRPC_OPTS,
    )
    worker_service_pb2_grpc.add_WorkerServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    servicer.set_server(server)
    servicer.start_tensor_server(port + servicer._tensor_port_offset)
    tag_str = f"  tags=[{', '.join(tags)}]" if tags else ""
    _LOG.info("worker started port=%s hostname=%s device=%s%s", port, _HOSTNAME, resolved_device, tag_str)

    # ── announce to coordinator (centralized topology only) ─────────────────
    if coordinator_addr:
        from ..discovery import NodeInfo, announce_to_coordinator
        node_info = NodeInfo(
            node_id=_HOSTNAME,
            address=worker_address,
            device=resolved_device,
            memory_mb=memory_mb,
            tags=tags,
        )
        _LOG.info("registering with coordinator address=%s", coordinator_addr)
        try:
            result = announce_to_coordinator(
                coordinator_addr,
                node_info,
                retry_policy=RetryPolicy(
                    max_attempts=cfg.discovery.registration_max_attempts,
                    delay_s=cfg.discovery.registration_delay_s,
                    rpc_timeout_s=cfg.discovery.registration_rpc_timeout_s,
                ),
            )
            _LOG.info("registration complete run_id=%s worker_index=%s", result.run_id, result.worker_index)
        except RuntimeError as e:
            _LOG.error("registration failed fatally: %s", e)
            server.stop(0)
            sys.exit(1)

        # ── coordinator watchdog ─────────────────────────────────────────────
        # Re-registers periodically so that:
        #   (a) a restarted coordinator finds this worker without container restart
        #   (b) workers are reusable across training jobs without manual intervention
        watchdog_interval = cfg.discovery.watchdog_interval_s
        _shutdown_watchdog = threading.Event()

        def _watchdog():
            while not _shutdown_watchdog.wait(timeout=watchdog_interval):
                try:
                    announce_to_coordinator(
                        coordinator_addr, node_info,
                        retry_policy=RetryPolicy(max_attempts=1, delay_s=0.0, rpc_timeout_s=5.0),
                    )
                except RuntimeError:
                    pass  # coordinator unreachable — try again next tick

        wt = threading.Thread(target=_watchdog, name="coordinator-watchdog", daemon=True)
        wt.start()

        server.wait_for_termination()
        _shutdown_watchdog.set()
    else:
        server.wait_for_termination()

    _LOG.info("worker terminated cleanly hostname=%s", _HOSTNAME)
