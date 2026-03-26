"""
WorkerServicer — gRPC service implementation for TorchSlicer workers.

Shared by centralized and P2P topologies. Centralised workers register
with a coordinator; P2P workers wait for init() from the driver node.
"""

import io
import socket
import threading
import traceback
from concurrent import futures

import torch
from torch import nn, optim

from ..core.split_layer import SplitLayer
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


def _channel(addr):
    return grpc.insecure_channel(addr, options=_GRPC_OPTS)


class _StaleWorkError(RuntimeError):
    """Raised when queued/in-flight work belongs to a superseded worker config."""


class _PeerUnavailableError(RuntimeError):
    """Raised when a neighboring worker becomes unreachable mid-batch."""


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


def deserialize_tensor(msg) -> torch.Tensor:
    dtype = _DTYPE_MAP.get(msg.dtype, torch.float32)
    return torch.frombuffer(bytearray(msg.data), dtype=dtype).reshape(list(msg.shape)).clone()


def serialize_tensor(t: torch.Tensor):
    return worker_service_pb2.Tensor(
        data=_tensor_to_bytes(t),
        shape=list(t.shape),
        dtype=_TORCH_TO_DTYPE.get(t.dtype, worker_service_pb2.FLOAT32),
    )


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
    return advertised or f"{host}:{port}"


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

        self._reset_state()

        # Single-worker compute pool: serialises forward/backward on this worker
        # so concurrent micro-batch RPCs never race on self.layer.x or .grad tensors.
        self._pool = futures.ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()

    def set_server(self, server):
        self._server = server

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
                self._next_stub = worker_service_pb2_grpc.WorkerServiceStub(
                    _channel(self.next_worker))
            if self.prev_worker:
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
                "worker init run_id=%s index=%s layers=%s is_last=%s device=%s prev=%s next=%s param_mb=%.2f cuda_alloc_mb=%.2f n_micro=%s profile_verbosity=%s profile_memory=%s",
                self._run_id,
                self._worker_index,
                layer_names,
                self.is_last,
                self.device,
                self.prev_worker,
                self.next_worker,
                param_mb,
                cuda_alloc_mb,
                self._n_micro,
                request.profile_verbosity,
                request.profile_memory,
            )

            with _tracer.span(
                "torchslicer.worker.init",
                worker=socket.gethostname(),
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
                ok=True, message="Initialized", hostname=socket.gethostname())
        except Exception as e:
            _LOG.exception("worker init failed: %s", e)
            return worker_service_pb2.StatusMessage(
                ok=False, message=str(e), hostname=socket.gethostname())

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

            _LOG.info("worker stopping hostname=%s", socket.gethostname())
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

            tensor = deserialize_tensor(request.input).to(self.device)
            # Unpack aux inputs (multimodal models — sent to first worker only)
            aux = {}
            if request.aux_inputs:
                aux = {k: deserialize_tensor(v).to(self.device)
                       for k, v in request.aux_inputs.items()}
            with _tracer.span(
                "torchslicer.worker.forward",
                batch_id=batch_id,
                worker=socket.gethostname(),
                input_shape=str(tuple(tensor.shape)),
            ) as s:
                with self._profiler.phase("forward"):
                    out   = self.layer(tensor, **aux)
                    x_ref = self.layer.x
                if s:
                    s.set_attribute("output_shape", str(tuple(out.shape)))

            with self._lock:
                self._ensure_generation(generation, "forward", batch_id)
                self._outputs[batch_id] = (out, x_ref)

            self._ensure_generation(generation, "forward", batch_id)
            with self._profiler.phase("send_fwd"):
                try:
                    self._next_stub.forward(worker_service_pb2.ForwardRequest(
                        batch_id=batch_id,
                        input=serialize_tensor(out),
                    ))
                except Exception as exc:
                    if self._is_unavailable_rpc(exc):
                        raise _PeerUnavailableError(
                            f"peer unavailable phase=forward batch_id={batch_id} peer={self.next_worker}"
                        ) from exc
                    raise

            self._profiler.mark_idle_start("bwd")

        except _StaleWorkError as e:
            _LOG.info("%s", e)
        except _PeerUnavailableError as e:
            _LOG.warning("%s", e)
        except Exception as e:
            self._report_fatal_error("forward", request.batch_id, e)

    def _forward_last(self, batch_id: int, request, generation: int):
        try:
            self._ensure_generation(generation, "forward_last", batch_id)
            self._before_runtime_phase("forward_last", batch_id)
            is_label = bool(request.label.data)

            if is_label:
                label = deserialize_tensor(request.label).to(self.device)
                with self._lock:
                    self._ensure_generation(generation, "forward_last", batch_id)
                    self._labels[batch_id] = label
                    cached = self._outputs.pop(batch_id, None)
                if cached is not None:
                    out, x_ref = cached
                    self._run_backward_last(batch_id, out, x_ref, label, generation)
            else:
                tensor = deserialize_tensor(request.input).to(self.device)
                with _tracer.span(
                    "torchslicer.worker.forward",
                    batch_id=batch_id,
                    worker=socket.gethostname(),
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

            grad_in = deserialize_tensor(request.gradient).to(self.device)

            with self._lock:
                self._ensure_generation(generation, "backward", batch_id)
                cached = self._outputs.pop(batch_id, None)

            if cached is None:
                _LOG.warning("backward skipped because cached output is missing for batch_id=%s", batch_id)
                return

            out, x_ref = cached

            with _tracer.span(
                "torchslicer.worker.backward",
                batch_id=batch_id,
                worker=socket.gethostname(),
                is_last=False,
            ):
                with self._profiler.phase("backward"):
                    # Pop MoE aux loss BEFORE backward — same graph nodes, would
                    # fail if popped after loss.backward() frees saved tensors.
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
            if is_last_micro and self.device.type == "cuda":
                torch.cuda.empty_cache()
            self._send_backward(batch_id, grad, is_last_micro, generation)
        except _StaleWorkError as e:
            _LOG.info("%s", e)
        except _PeerUnavailableError as e:
            _LOG.warning("%s", e)
        except Exception as e:
            self._report_fatal_error("backward", request.batch_id, e)

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
                self._coord_stub.report_metrics(coordinator_service_pb2.MetricsMessage(
                    batch_id=batch_id,
                    loss=avg_loss,
                    worker=socket.gethostname(),
                    run_id=self._run_id,
                ))

            with _tracer.span(
                "torchslicer.worker.backward",
                batch_id=batch_id,
                worker=socket.gethostname(),
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
            if is_last_micro and self.device.type == "cuda":
                torch.cuda.empty_cache()
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
        if self._prev_stub:
            with self._profiler.phase("send_bwd"):
                try:
                    self._prev_stub.backward(worker_service_pb2.BackwardRequest(
                        batch_id=batch_id,
                        gradient=serialize_tensor(grad),
                    ))
                except Exception as exc:
                    if self._is_unavailable_rpc(exc):
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
                        worker=socket.gethostname(),
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
        elif is_last_micro:
            self._coord_stub.batch_done(coordinator_service_pb2.BatchDoneRequest(
                batch_id=batch_id,
                run_id=self._run_id,
            ))
            self._profiler.mark_idle_start("fwd")
            self._profiler.end_batch()

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
    coordinator_addr=None,
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
                          Env: ``COORDINATOR_ADDRESS``. Pass ``None`` (or leave
                          ``COORDINATOR_ADDRESS`` unset) to skip registration —
                          required for P2P followers that wait for ``init()``
                          from the driver instead of self-registering.
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
    configure_process_logging(os.environ.get("LOG_LEVEL"))

    # ── resolve args / env vars ─────────────────────────────────────────────
    if port is None:
        port = int(os.environ.get("PORT", 50051))
    port = int(port)

    # coordinator_addr=None → skip registration (P2P followers, static-discovery)
    if coordinator_addr is None:
        coordinator_addr = os.environ.get("COORDINATOR_ADDRESS") or None

    hostname = socket.gethostname()
    if worker_address is None:
        worker_address = resolve_worker_address(port, hostname=hostname)

    if device is not None:
        os.environ["DEVICE"] = device
    resolved_device = resolve_device()
    memory_mb       = get_available_memory_mb(resolved_device)

    # WORKER_TAGS=gpu,high-memory  →  ["gpu", "high-memory"]
    raw_tags = os.environ.get("WORKER_TAGS", "").strip()
    tags = [t.strip() for t in raw_tags.split(",") if t.strip()] if raw_tags else []

    # ── build servicer + gRPC server ────────────────────────────────────────
    if servicer_class is None:
        servicer_class = WorkerServicer
    servicer = servicer_class()

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=grpc_workers),
        options=_GRPC_OPTS,
    )
    worker_service_pb2_grpc.add_WorkerServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    servicer.set_server(server)
    tag_str = f"  tags=[{', '.join(tags)}]" if tags else ""
    _LOG.info("worker started port=%s hostname=%s device=%s%s", port, hostname, resolved_device, tag_str)

    # ── announce to coordinator (centralized topology only) ─────────────────
    if coordinator_addr:
        from ..discovery import NodeInfo, announce_to_coordinator
        node_info = NodeInfo(
            node_id=hostname,
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
                    max_attempts=int(os.environ.get(
                        "DISCOVERY_REGISTRATION_MAX_ATTEMPTS", "30")),
                    delay_s=float(os.environ.get(
                        "DISCOVERY_REGISTRATION_DELAY", "3.0")),
                    rpc_timeout_s=float(os.environ.get(
                        "DISCOVERY_REGISTRATION_RPC_TIMEOUT", "5.0")),
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
        watchdog_interval = float(os.environ.get("DISCOVERY_WATCHDOG_INTERVAL", "15"))
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

    _LOG.info("worker terminated cleanly hostname=%s", hostname)
