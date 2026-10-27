"""
WorkerServicer — gRPC service implementation for TorchSlicer workers.

Shared by centralized and P2P topologies. Centralised workers register
with a coordinator; P2P workers wait for init() from the driver node.
"""

import contextvars
import datetime
import io
import os
import pickle
import socket
import struct
import threading
import traceback
import warnings
from concurrent import futures
from multiprocessing.reduction import ForkingPickler

import torch
from torch import nn, optim

from ..config import NCCL_STORE_PORT_OFFSET as _NCCL_STORE_PORT_OFFSET, RunConfig
from ..core.determinism import set_deterministic
from ..core.split_layer import SplitLayer
from ..core.tied import optimizer_parameters, retie
from ..sl.hooks import CutContext, pack_payload, unpack_payload, tensor_nbytes

_UNSET = object()  # sentinel: distinguish "not provided" from explicit None
from ..monitor import tracer as _tracer
from ..monitor import WorkerProfiler
from ..monitor.process_logger import get_logger, configure as configure_process_logging
from ..retry import RetryPolicy
from ..transport.tunnel import Router

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

# How long a tied-parameter owner waits for the replicas' gradients before failing.
_TIED_GRAD_TIMEOUT_S = 300.0
# TCP tensor frame format (big-endian):
#   kind(1c) | batch_id(4I) | dtype(1B) | ndim(2H) | ctx_len(2H)
#   followed by: ctx_bytes(ctx_len) | shape(ndim×8q) | payload_len(8Q) | payload
# ctx_len=0 when tracing is disabled — the two header bytes are the only overhead.
_FRAME_HEADER = struct.Struct("!cIBHH")
_SHAPE_ITEM = struct.Struct("!q")
_PAYLOAD_LEN = struct.Struct("!Q")
_HOSTNAME = socket.gethostname()

# IPC frame (CUDA shared-memory handle exchange over TCP):
#   kind(1c) | batch_id(4I) | handle_len(4I) | ctx_len(2H)
#   followed by: ctx_bytes(ctx_len) | handle_bytes(handle_len)
# kind "f" = forward via IPC, "b" = backward via IPC (lowercase distinguishes from "F"/"B").
# After the frame the receiver sends a 1-byte ACK so the sender can release its IPC reference.
_IPC_FRAME_HEADER = struct.Struct("!cIIH")

# Rest-of-frame structs read after the 1-byte kind (used by _handle_tensor_conn dispatch):
_TCP_REST_HEADER = struct.Struct("!IBHH")   # batch_id(4I) dtype(1B) ndim(2H) ctx_len(2H)
_IPC_REST_HEADER = struct.Struct("!IIH")    # batch_id(4I) handle_len(4I) ctx_len(2H)

# Set TCP_NO_PINNED=1 to disable pinned-memory async D2H (falls back to the old
# synchronous .cpu().contiguous() path — useful for benchmarking the improvement).
_TCP_NO_PINNED = os.environ.get("TCP_NO_PINNED", "0") == "1"


class _StaleWorkError(RuntimeError):
    """Raised when queued/in-flight work belongs to a superseded worker config."""


class _PeerUnavailableError(RuntimeError):
    """Raised when a neighboring worker becomes unreachable mid-batch."""


class _TensorStreamError(RuntimeError):
    """Raised when the raw tensor transport fails."""


class _TensorStreamClosed(_TensorStreamError):
    """Raised when the raw tensor stream is closed cleanly by the peer."""


class _TensorDecodeError(_TensorStreamError):
    """A frame arrived but its tensor could not be rebuilt (e.g. a CUDA IPC handle that
    does not open here): the receiver reports it, since the sender only sees a closed
    stream and the heartbeats see a live peer."""

    def __init__(self, batch_id: int, exc: Exception):
        super().__init__(f"cannot rebuild the tensor of batch {batch_id}: {exc}")
        self.batch_id = batch_id


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
        (worker_service_pb2.UINT8,    torch.uint8),
        (worker_service_pb2.INT8,     torch.int8),
        (worker_service_pb2.BOOL,     torch.bool),
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


def deserialize_tensor(msg, device="cpu") -> torch.Tensor:
    """``msg.data`` is already a fresh copy (upb copies on access): view it and let the
    one ``.to`` be the only other copy, straight to ``device``. Each fresh buffer of
    32 MB or more page-faults (glibc mmap threshold), so every copy saved counts."""
    dtype = _DTYPE_MAP.get(msg.dtype, torch.float32)
    shape = list(msg.shape)
    data = msg.data
    if not data:
        return torch.empty(shape, dtype=dtype, device=device)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The given buffer is not writable")
        view = torch.frombuffer(data, dtype=dtype)
    return view.reshape(shape).to(device, copy=True)   # bytes are immutable: always copy


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


def _recv_into(sock: socket.socket, buf: bytearray, idle_timeout_ok: bool = False) -> memoryview:
    """Fill ``buf`` from ``sock``.

    The socket has a short timeout so the reader can poll its stop flag.  That
    timeout may only surface while idle *between* frames (``idle_timeout_ok`` and
    nothing read yet); inside a frame a slow sender must not desynchronise the
    stream, so timeouts there just retry.
    """
    view = memoryview(buf)
    offset = 0
    while offset < len(view):
        try:
            n_recv = sock.recv_into(view[offset:])
        except socket.timeout:
            if idle_timeout_ok and offset == 0:
                raise
            continue
        if n_recv == 0:
            if offset == 0:
                raise _TensorStreamClosed("peer closed tensor stream")
            raise _TensorStreamError("unexpected EOF while reading tensor frame")
        offset += n_recv
    return view


def _load_transport_settings() -> tuple[str, int]:
    cfg = RunConfig.load(os.environ.get("EXPERIMENT_CONFIG"))
    tensor_transport = cfg.transport.tensor.strip().lower()
    if tensor_transport not in {"grpc", "tcp", "ipc", "nccl"}:
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
    kind, batch_id, dtype_code, ndim, ctx_len = _FRAME_HEADER.unpack(
        _recv_into(sock, header_buf, idle_timeout_ok=True))
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
    """TCP peer client with pinned-memory D2H for CUDA tensors.

    On CUDA: copies the tensor into a per-connection pinned host buffer with
    ``non_blocking=True`` (DMA), records a CUDA event, builds the frame header
    in parallel, then synchronises the event right before ``sendall``.  The old
    path (``.cpu()``) allocates fresh host memory per send, which page-faults
    above glibc's 32 MB mmap threshold: 20 ms vs 1.4 ms for a 33.6 MB tensor.
    ``TCP_NO_PINNED=1`` forces the old path (benchmarks).
    On CPU (or when CUDA is unavailable): falls back to the original path.
    The pinned buffer grows with a 25 % headroom (high-water-mark); it is never
    shrunk and is allocated at most once per peer connection per run.
    """

    def __init__(self, address: str):
        self.address = address
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._pinned: torch.Tensor | None = None   # uint8, pinned, high-water-mark
        self._d2h_event: torch.cuda.Event | None = None

    def _get_pinned(self, nbytes: int) -> torch.Tensor:
        if self._pinned is None or self._pinned.numel() < nbytes:
            alloc = max(nbytes, int(nbytes * 1.25))
            self._pinned = torch.empty(alloc, dtype=torch.uint8, pin_memory=True)
        return self._pinned[:nbytes]

    def send_tensor(self, kind: bytes, batch_id: int, tensor: torch.Tensor) -> None:
        if torch.cuda.is_available() and tensor.is_cuda and not _TCP_NO_PINNED:
            self._send_pinned(kind, batch_id, tensor)
        else:
            self._send_cpu(kind, batch_id, tensor)

    def _send_pinned(self, kind: bytes, batch_id: int, tensor: torch.Tensor) -> None:
        arr = tensor.detach().contiguous()          # keep on GPU, ensure contiguous
        nbytes = arr.nbytes
        buf = self._get_pinned(nbytes)
        # Kick off async DMA: GPU → pinned host memory (does NOT block the GPU stream).
        buf.copy_(arr.view(torch.uint8).flatten(), non_blocking=True)
        # Record a CUDA event on the current stream; we'll sync it right before sendall.
        if self._d2h_event is None:
            self._d2h_event = torch.cuda.Event()
        self._d2h_event.record()
        # Build the frame header while the DMA runs (pure CPU work).
        shape = list(arr.shape)
        ctx_bytes = _tracer.inject_context_bytes()
        header = _FRAME_HEADER.pack(kind, int(batch_id), _dtype_to_code(arr.dtype),
                                     len(shape), len(ctx_bytes))
        shape_bytes = b"".join(_SHAPE_ITEM.pack(int(d)) for d in shape)
        payload_len_bytes = _PAYLOAD_LEN.pack(nbytes)
        meta = header + ctx_bytes + shape_bytes + payload_len_bytes
        # Synchronise: wait until DMA has written every byte to pinned memory.
        self._d2h_event.synchronize()
        with self._lock:
            self._ensure_connected()
            try:
                self._sock.sendall(meta)
                self._sock.sendall(buf.numpy())     # send directly from pinned memory
            except Exception as exc:
                self.close()
                raise _TensorStreamError(f"tensor send failed peer={self.address}") from exc

    def _send_cpu(self, kind: bytes, batch_id: int, tensor: torch.Tensor) -> None:
        header, ctx_bytes, shape_bytes, payload_len, payload = _pack_tensor_frame(kind, batch_id, tensor)
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


def _is_same_machine(addr: str) -> bool:
    """Return True if ``addr`` (host:port) resolves to the local machine."""
    host = addr.rsplit(":", 1)[0].strip("[]")  # strip IPv6 brackets
    if host in ("localhost", "127.0.0.1") or host.startswith("127."):
        return True
    try:
        peer_ip = socket.gethostbyname(host)
        local_ips = {info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None)}
        local_ips.add("127.0.0.1")
        return peer_ip in local_ips
    except Exception:
        return False


_CUDA_IPC: bool | None = None


def _cuda_ipc_supported() -> bool:
    """Whether CUDA IPC works here: a discrete GPU. Integrated ones (Jetson/Tegra) don't
    have it (the Orin Nano refuses the export, the AGX Orin exports a handle that the
    other process can't open), so ipc falls back to TCP there."""
    global _CUDA_IPC
    if _CUDA_IPC is None:
        _CUDA_IPC = torch.cuda.is_available() and not torch.cuda.get_device_properties(
            torch.cuda.current_device()).is_integrated
        if torch.cuda.is_available() and not _CUDA_IPC:
            _LOG.warning("transport.tensor=ipc: no CUDA IPC on an integrated GPU (%s); using tcp",
                         torch.cuda.get_device_name())
    return _CUDA_IPC


class _IpcPeerClient:
    """CUDA IPC peer client for same-machine GPU-to-GPU tensor transfer.

    Serialises the tensor as a CUDA IPC handle (a ~350-byte descriptor) via
    ``ForkingPickler``, sends it over the existing TCP connection, and waits for
    a 1-byte ACK from the receiver before returning.  The receiver reconstructs
    the tensor from the handle without any host copy and sends the ACK after
    cloning the result.

    Falls back to ``_TensorPeerClient`` (pinned TCP) for CPU tensors or when the
    peers are on different machines (guard in the factory code in ``init``).
    """

    def __init__(self, address: str, fallback: _TensorPeerClient):
        self.address = address
        self._fallback = fallback
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def send_tensor(self, kind: bytes, batch_id: int, tensor: torch.Tensor) -> None:
        if not (torch.cuda.is_available() and tensor.is_cuda):
            self._fallback.send_tensor(kind, batch_id, tensor)
            return

        arr = tensor.detach().contiguous()
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(arr)
        handle_bytes = buf.getvalue()

        ctx_bytes = _tracer.inject_context_bytes()
        # Lowercase kind signals IPC to the receiver ("f" = forward, "b" = backward).
        ipc_kind = kind.lower()
        header = _IPC_FRAME_HEADER.pack(ipc_kind, int(batch_id), len(handle_bytes), len(ctx_bytes))
        frame = header + ctx_bytes + handle_bytes

        with self._lock:
            self._ensure_connected()
            try:
                self._sock.sendall(frame)
                # Block until receiver has cloned the tensor and released the IPC ref.
                ack = _recv_exact(self._sock, 1)
                if ack != b"\x00":
                    raise _TensorStreamError(f"IPC ACK mismatch peer={self.address}")
            except Exception as exc:
                self.close()
                raise _TensorStreamError(f"IPC tensor send failed peer={self.address}") from exc

    def _ensure_connected(self) -> None:
        if self._sock is not None:
            return
        host, port = self.address.rsplit(":", 1)
        sock = socket.create_connection((host, int(port)), timeout=10.0)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock

    def close(self) -> None:
        for client in (self, self._fallback):
            if client._sock is not None:
                try:
                    client._sock.close()
                except Exception:
                    pass
                client._sock = None


class _RecvBuf:
    """Per-connection high-water-mark receive buffer.

    Re-uses the same ``bytearray`` across frames to avoid per-batch allocation.
    Grows as needed (25 % headroom); never shrinks.  Call ``get(n)`` to obtain a
    ``memoryview`` of exactly ``n`` writable bytes; the view is valid until the
    next ``get()`` call.
    """
    __slots__ = ("_arr",)

    def __init__(self):
        self._arr: bytearray | None = None

    def get(self, n: int) -> memoryview:
        if self._arr is None or len(self._arr) < n:
            self._arr = bytearray(max(n, int(n * 1.25)))
        return memoryview(self._arr)[:n]


def _read_tensor_frame_rest(
    sock: socket.socket, recv_buf: _RecvBuf
) -> tuple[int, torch.Tensor, bytes]:
    """Read the rest of a regular TCP tensor frame after the 1-byte kind has been read."""
    rest_buf = bytearray(_TCP_REST_HEADER.size)
    _recv_into(sock, rest_buf)
    batch_id, dtype_code, ndim, ctx_len = _TCP_REST_HEADER.unpack(rest_buf)
    ctx_bytes = bytes(_recv_into(sock, bytearray(ctx_len))) if ctx_len else b""
    shape_buf = bytearray(_SHAPE_ITEM.size * ndim)
    shape_view = _recv_into(sock, shape_buf)
    shape = [_SHAPE_ITEM.unpack_from(shape_view, i * _SHAPE_ITEM.size)[0] for i in range(ndim)]
    payload_len_buf = bytearray(_PAYLOAD_LEN.size)
    payload_len = _PAYLOAD_LEN.unpack(_recv_into(sock, payload_len_buf))[0]
    buf = recv_buf.get(payload_len)
    _recv_into(sock, buf)
    dtype = _code_to_dtype(dtype_code)
    # clone() so the caller can reuse recv_buf for the next frame
    tensor = torch.frombuffer(buf, dtype=dtype).reshape(shape).clone()
    return batch_id, tensor, ctx_bytes


def _read_ipc_frame_rest(sock: socket.socket) -> tuple[int, torch.Tensor, bytes]:
    """Read the rest of an IPC frame after the 1-byte kind, reconstruct the tensor, send ACK."""
    rest_buf = bytearray(_IPC_REST_HEADER.size)
    _recv_into(sock, rest_buf)
    batch_id, handle_len, ctx_len = _IPC_REST_HEADER.unpack(rest_buf)
    ctx_bytes = bytes(_recv_into(sock, bytearray(ctx_len))) if ctx_len else b""
    handle_buf = bytearray(handle_len)
    _recv_into(sock, handle_buf)
    # Reconstruct the CUDA tensor from the IPC handle; clone immediately so the
    # sender can safely release its allocation after receiving the ACK.
    try:
        tensor = pickle.loads(handle_buf).clone()
    except Exception as exc:
        raise _TensorDecodeError(batch_id, exc) from exc
    sock.sendall(b"\x00")           # ACK: sender may now release its IPC reference
    return batch_id, tensor, ctx_bytes


def _indexed(device: torch.device) -> torch.device:
    """``device`` with an explicit index (``cuda`` → ``cuda:<current>``)."""
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def resolve_device(value: str | None = None) -> str:
    """Resolve the target device from ``value``, else the DEVICE environment variable.

    DEVICE=auto  (default) — use ``cuda`` if available, else ``mps`` if
                             available, else ``cpu``
    DEVICE=cuda            — prefer CUDA; falls back to ``cpu`` when CUDA is
                             unavailable
    DEVICE=mps             — prefer Apple Metal (MPS); falls back to ``cpu``
                             when MPS is unavailable
    DEVICE=cpu             — always use CPU regardless of hardware
    """
    import os
    val = (value or os.environ.get("DEVICE", "auto")).lower().strip()

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


def measure_compute_gflops(device, n: int | None = None, repeats: int = 5) -> float:
    """Median fp32 matmul throughput (GFLOP/s) of ``device``: a coarse speed for planning.

    Uses constant matrices, so the global RNG (and with it data order and init) is
    untouched.
    """
    import statistics
    import time
    dev = torch.device(device)
    n = n or (2048 if dev.type == "cuda" else 384)
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    a = torch.full((n, n), 0.5, device=dev)
    b = torch.full((n, n), 0.25, device=dev)
    with torch.no_grad():
        for _ in range(2):
            a @ b
        sync()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            a @ b
            sync()
            times.append(time.perf_counter() - t0)
    return 2 * n ** 3 / max(statistics.median(times), 1e-9) / 1e9


def measure_link(stub, n_bytes: int = 1 << 20, repeats: int = 3, timeout: float = 30.0):
    """(rtt_ms, bandwidth_mbps) to the worker behind ``stub``, using its ``echo`` RPC:
    the median time of an empty echo, and ``n_bytes`` over the extra time a payload of
    that size takes."""
    import statistics
    import time

    def _time(payload: bytes) -> float:
        out = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            stub.echo(worker_service_pb2.EchoRequest(payload=payload), timeout=timeout)
            out.append(time.perf_counter() - t0)
        return statistics.median(out)

    _time(b"")                                   # connect
    rtt = _time(b"")
    data = _time(bytes(n_bytes)) if n_bytes else rtt
    mbps = n_bytes * 8 / max(data - rtt, 1e-6) / 1e6 if n_bytes else 0.0
    return rtt * 1e3, mbps


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
        self._persistent = False
        self._tensor_server = None
        self._tensor_stop = threading.Event()
        self._tensor_threads: list[threading.Thread] = []
        # Declared capabilities (network.worker_* config); measured ones fill the gaps.
        self.declared = {"compute_gflops": 0.0, "max_memory_mb": 0, "first_slice": False, "tags": []}
        self._measured_gflops = None
        # In-process coordinator (P2P driver): called directly instead of over gRPC.
        # Anything with the CoordinatorService stub methods works.
        self.local_coordinator = None
        self.data = None          # a client node's loader (sl.client_nodes); kept across runs
        # Where calls to other nodes go (transport/tunnel.py): direct, by tunnel, or
        # through a relay. ``nat``: this worker accepts no connections, so it opens a
        # tunnel to each neighbour at init (the coordinator's is opened at start).
        self.router = Router()
        self.nat = False
        self._peer_tunnels: set = set()

        self._tied_cond = threading.Condition()
        self._data_lock = threading.Lock()
        self._round_done: dict = {}   # client_round: batch -> Event, set when its backward is done
        self._dist_group  = None  # torch.distributed ProcessGroup for NCCL transport; None otherwise
        self._nccl_future = None  # pending rendezvous Future; waited on before first NCCL transfer
        self._nccl_meta   = futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="nccl-meta")
        self._label_calls = futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="labels")
        self._reset_state()

        # Single-worker compute pool: serialises forward/backward on this worker
        # so concurrent micro-batch RPCs never race on self.layer.x or .grad tensors.
        self._pool = futures.ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()

    def set_server(self, server):
        self._server = server

    def start_tensor_server(self, port: int):
        if self._tensor_transport not in {"tcp", "ipc"}:
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
        for peer in self._tensor_peers():
            peer.close()

    def _tensor_peers(self) -> list:
        """The raw TCP / IPC clients to this worker's neighbours (every client node's, on
        a slice 1 that serves several)."""
        peers = [self._next_tensor, self._prev_tensor, *(t for _, _, t in self._prev_peers)]
        return [p for p in peers if p is not None]

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
        recv_buf = _RecvBuf()
        try:
            while not self._tensor_stop.is_set():
                # Read the 1-byte kind field; this is the only idle-timeout point.
                kind_buf = bytearray(1)
                try:
                    _recv_into(conn, kind_buf, idle_timeout_ok=True)
                except socket.timeout:
                    continue
                kind = kind_buf
                generation = self._generation

                if kind in (b"F", b"B"):
                    batch_id, tensor, ctx_bytes = _read_tensor_frame_rest(conn, recv_buf)
                    with _tracer.extract_context(ctx_bytes):
                        if kind == b"F":
                            self._submit_forward(batch_id, self._forward_tensor,
                                                 batch_id, tensor, generation)
                        else:
                            self._pool.submit(
                                _tracer.propagate_to_thread(self._backward_tensor),
                                batch_id, tensor, generation,
                            )
                elif kind in (b"f", b"b"):
                    # CUDA IPC frame: reconstruct GPU tensor from handle, send ACK inside helper.
                    batch_id, tensor, ctx_bytes = _read_ipc_frame_rest(conn)
                    with _tracer.extract_context(ctx_bytes):
                        if kind == b"f":
                            self._submit_forward(batch_id, self._forward_tensor,
                                                 batch_id, tensor, generation)
                        else:
                            self._pool.submit(
                                _tracer.propagate_to_thread(self._backward_tensor),
                                batch_id, tensor, generation,
                            )
                else:
                    raise _TensorStreamError(f"unknown tensor frame kind={kind!r}")
        except _TensorDecodeError as exc:
            self._report_fatal_error("ipc_receive", exc.batch_id, exc)
        except Exception as exc:
            if not self._tensor_stop.is_set() and not isinstance(exc, _TensorStreamClosed):
                _LOG.warning("tensor transport connection closed addr=%s error=%s", addr, exc)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _reset_state(self, keep_data: bool = False):
        """Drop the slice. ``keep_data``: a client node keeps its place in its data (a re-init
        within the same run, e.g. after a server worker's recovery, then replays the epoch
        from the same random state)."""
        if not keep_data:
            self._data_epoch = None   # a client node: the epoch its loader iterator is on
            self._data_rng = None     # ... and the random state that epoch started from
            self._data_iter = None
            self._data_index = 0
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
        self._prev_peers: list = []   # [(address, stub, tensor client)] by client index (M10)
        self._gate_batch = None       # _gated(): the batch in flight on this slice
        self._gate_parked: list = []  # ... and forwards of later batches, in arrival order
        self._label_stub = None       # a client node: the last slice, for its labels
        self._n_clients = 0           # a server slice with client nodes: how many
        self._per_client = False      # SplitFed V1: one copy of this slice per client
        self._client_copies: dict = {}
        self._active_client = None    # whose copy is in the slice now
        self._round_initial = None    # the slice at the round's start (a new client's copy)
        self._coord_stub = None
        self._fatal_error = None

        self._labels:       dict = {}
        self._outputs:      dict = {}
        self._micro_losses: dict = {}

        self._profiler: WorkerProfiler = WorkerProfiler(verbosity=0)

        # tied parameters (see core/tied.py): [{group, param, owner, stubs}]
        self._tied:        list = []
        self._tied_token:  str  = ""
        self._tied_grads:  dict = {}   # (group, batch_id) → [grad from each replica]

        # cut hooks on this slice's input cut (index - 1) and output cut (index)
        self._hooks:    dict = {}
        self._n_slices: int  = 0
        self._reset_comm()

    def _reset_comm(self):
        """Communication sent and compute time since the last epoch read (get_stats)."""
        self._comm = {"bytes_fwd": 0, "bytes_bwd": 0, "raw_bytes_fwd": 0, "raw_bytes_bwd": 0,
                      "msgs_fwd": 0, "msgs_bwd": 0, "bytes_label": 0, "msgs_label": 0}
        self._compute_s = 0.0

    # ── init ───────────────────────────────────────────────────────────────────

    def init(self, request, context):
        try:
            with self._lock:
                self._generation += 1
                self._reset_state(keep_data=bool(request.run_id) and request.run_id == self._run_id)

            if request.deterministic:
                set_deterministic()
            layers = self._build_layers(request.layers)
            predecessors = (
                [list(p.indices) for p in request.predecessors]
                if request.predecessors else None
            )
            self.layer   = SplitLayer(layers, is_last=request.is_last, predecessors=predecessors)
            self.is_last = request.is_last
            self._run_id       = request.run_id
            self._worker_index = request.worker_index
            self._n_slices     = request.n_slices
            self._hooks = (torch.load(io.BytesIO(request.cut_hooks), weights_only=False)
                           if request.cut_hooks else {})

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
            self._configure_tied(request.tied, request.init_token)
            params = optimizer_parameters(
                self.layer, exclude={n for t in request.tied if not t.owner for n in t.names})
            # extra_params holds the user's kwargs exactly (lr included when given), so the
            # optimizer is the one the local executor builds; the float32 ``lr`` field is
            # only a fallback for a driver that sends no extra_params.
            opt_kwargs = opt_extra if request.optimizer.extra_params else {"lr": request.optimizer.lr}
            opt = getattr(optim, request.optimizer.name)(params, **opt_kwargs) if params else None
            self.layer.set_optimizer(opt)

            self.prev_worker = request.prev_worker or None
            self.next_worker = request.next_worker or None
            self.coordinator = request.coordinator
            self._n_micro    = max(1, request.n_micro) if request.n_micro else 1

            # gRPC stubs always (a peer behind a tunnel or relay is reached through
            # them); raw TCP is used for a peer while it is directly reachable.
            channel = self.router.channel
            if self.next_worker:
                self._next_stub, self._next_tensor = self._peer_links(self.next_worker)
            if self.prev_worker:
                self._prev_stub, self._prev_tensor = self._peer_links(self.prev_worker)
            if request.prev_workers:
                if self._tensor_transport == "nccl":
                    raise ValueError("client nodes (sl.client_nodes) don't support "
                                     "transport.tensor=nccl: its process group is fixed at init")
                self._prev_peers = self._client_links(request.prev_workers)
            if request.label_worker:
                self._label_stub = worker_service_pb2_grpc.WorkerServiceStub(
                    channel(request.label_worker))
            self._n_clients = int(request.n_clients)
            self._per_client = bool(request.per_client_state)
            if self._per_client and not self._n_clients:
                raise ValueError("per_client_state needs n_clients")
            if self.local_coordinator is not None:
                self._coord_stub = self.local_coordinator
            elif self.coordinator:
                self._coord_stub = coordinator_service_pb2_grpc.CoordinatorServiceStub(
                    channel(self.coordinator))
            if self.nat:
                self._open_peer_tunnels(request)

            # NCCL ProcessGroup: all workers rendezvous on a TCPStore the coordinator
            # hosts at coordinator_port + _NCCL_STORE_PORT_OFFSET.
            if self._tensor_transport == "nccl" and request.n_slices > 0:
                self._init_nccl_group(
                    request.coordinator, request.worker_index, request.n_slices,
                    request.init_token,
                )

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

    def _peer_links(self, address: str) -> tuple:
        """``(gRPC stub, raw tensor client or None)`` to the neighbour at ``address``."""
        stub = worker_service_pb2_grpc.WorkerServiceStub(self.router.channel(address))
        if self._tensor_transport not in {"tcp", "ipc"}:
            return stub, None
        tcp_client = _TensorPeerClient(_tensor_addr(address, self._tensor_port_offset))
        if self._tensor_transport == "ipc" and _is_same_machine(address) and _cuda_ipc_supported():
            return stub, _IpcPeerClient(_tensor_addr(address, self._tensor_port_offset), tcp_client)
        return stub, tcp_client

    def _client_links(self, addresses) -> list:
        """``[(address, stub, tensor client)]`` per client; ``""`` = no client node there yet."""
        return [(a, *self._peer_links(a)) if a else ("", None, None) for a in addresses]

    def set_peers(self, request, context):
        """Slice 1 with client nodes: the clients' addresses now (a client joined, or came
        back at another address). Runs on the compute pool, between batches."""
        def _update():
            old = self._prev_peers
            self._prev_peers = self._client_links(request.prev_workers)
            for _, _, t in old:          # a client back at the same address is a new process
                if t is not None:
                    t.close()
            if self.nat and not self.router.relay:
                for a in request.prev_workers:
                    if a:
                        self.router.keep(a)
        self._pool.submit(_update).result()
        return worker_service_pb2.Ack(batch_id=0)

    def _prev_for(self, batch_id: int) -> tuple:
        """``(address, stub, tensor client)`` of the slice the gradient of ``batch_id`` goes
        back to: the previous worker, or on a slice 1 serving client nodes, the client that
        sent the batch, ``(batch_id // n_micro) % n_clients``."""
        if self._prev_peers:
            return self._prev_peers[(batch_id // self._n_micro) % len(self._prev_peers)]
        return self.prev_worker, self._prev_stub, self._prev_tensor

    def _init_nccl_group(self, coordinator: str, rank: int, world_size: int,
                         token: str = "") -> None:
        """Kick off NCCL ProcessGroup init in a background thread.

        init_process_group blocks until all world_size ranks have joined.
        Calling it synchronously inside init() deadlocks when the coordinator
        sends SliceConfigs sequentially: rank 0's init() would wait for rank 1
        to join, but rank 1 hasn't received its SliceConfig yet.

        Instead we launch the rendezvous in a thread and store a Future;
        _wait_nccl() is called before the first NCCL send/recv. The outcome is
        written to the store under ``ts/nccl/<token>/status/<rank>`` ("ok" or
        "err:<reason>"), which the coordinator waits on after init
        (``DistributedExecutor._wait_nccl_ready``), so a failure stops the run
        with its reason instead of hanging the first transfer.
        """
        self._nccl_future: futures.Future | None = None
        executor = futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="nccl-init")

        def _rendezvous():
            import torch.distributed as dist
            coord_host, coord_port_str = coordinator.rsplit(":", 1)
            nccl_port = int(coord_port_str) + _NCCL_STORE_PORT_OFFSET
            store = dist.TCPStore(
                host_name=coord_host,
                port=nccl_port,
                world_size=world_size,
                is_master=False,
                timeout=datetime.timedelta(seconds=60),
            )
            # one key space per init (recovery re-inits on the same store)
            prefix = f"ts/nccl/{token}"
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
                cuda = self.device.type == "cuda"
                if cuda:
                    # NCCL allows one rank per GPU; with two it fails only at the first
                    # transfer ("Duplicate GPU detected"). Check here, with a clear reason.
                    gpu = f"{socket.gethostname()}/{torch.cuda.get_device_properties(self.device).uuid}"
                    store.set(f"{prefix}/gpu/{rank}", gpu)
                    keys = [f"{prefix}/gpu/{r}" for r in range(world_size)]
                    store.wait(keys)
                    owners = [store.get(k).decode() for k in keys]
                    dup = [r for r, g in enumerate(owners) if g == gpu and r != rank]
                    if dup:
                        raise RuntimeError(
                            f"transport.tensor=nccl needs one GPU per slice, but slices "
                            f"{sorted([rank, *dup])} are on the same GPU ({gpu}); use "
                            f"distinct GPUs, or transport.tensor=ipc on one GPU")
                dist.init_process_group(
                    backend="nccl" if cuda else "gloo",
                    store=dist.PrefixStore(f"{prefix}/pg", store),
                    rank=rank,
                    world_size=world_size,
                    # eager NCCL communicator: init errors surface here, not on first send;
                    # device_id needs an index ("cuda" alone is refused)
                    **({"device_id": _indexed(self.device)} if cuda else {}),
                )
            except Exception as exc:
                store.set(f"{prefix}/status/{rank}", f"err:{exc}")
                raise
            store.set(f"{prefix}/status/{rank}", "ok")
            return dist.group.WORLD

        self._nccl_future = executor.submit(_rendezvous)
        executor.shutdown(wait=False)
        _LOG.info("NCCL rendezvous started rank=%s world=%s (background)", rank, world_size)

    def _nccl_send(self, tensor: torch.Tensor, dst: int, metadata_call) -> None:
        """Send ``tensor`` to rank ``dst`` with NCCL, and the gRPC call that tells the peer
        its shape (the peer posts ``dist.recv`` when that call arrives). The call is
        started first, on another thread: NCCL connects a pair of ranks at their first
        transfer, and ``isend`` blocks until the peer joins, i.e. until it has the call."""
        import torch.distributed as dist
        call = self._nccl_meta.submit(contextvars.copy_context().run, metadata_call)
        try:
            dist.isend(tensor, dst=dst, group=self._dist_group).wait()
        finally:
            call.result()

    def _wait_nccl(self) -> bool:
        """Block until the background NCCL rendezvous completes. A failed rendezvous
        raises: falling back to gRPC on one worker only would leave its peers blocked
        in a NCCL recv that never comes."""
        if self._nccl_future is None:
            return False
        try:
            self._dist_group = self._nccl_future.result(timeout=120)
        except Exception as exc:
            raise RuntimeError(f"NCCL process group init failed: {exc}") from exc
        finally:
            self._nccl_future = None
        _LOG.info("NCCL group ready device=%s", self.device)
        return True

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

            if self._persistent:
                # End of run only: drop the slice (bumping the generation discards
                # in-flight work of the finished run) and wait for the next init().
                with self._lock:
                    self._generation += 1
                    for peer in self._tensor_peers():
                        peer.close()
                    self._reset_state()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _LOG.info("run ended, worker persistent — waiting for next run hostname=%s", _HOSTNAME)
                return

            _LOG.info("worker stopping hostname=%s", _HOSTNAME)
            self.stop_tensor_server()
            if self._server:
                self._server.stop(grace=1)

        threading.Thread(target=_do_shutdown, daemon=True).start()
        return worker_service_pb2.Ack(batch_id=0)

    # ── tied parameters ────────────────────────────────────────────────────────

    def _configure_tied(self, tied: list, init_token: str) -> None:
        """Re-tie shared parameters (layers arrive serialized one by one) and connect
        to the workers holding other copies."""
        self._tied_token = init_token
        self._tied = [
            {
                "group": t.group,
                "param": retie(self.layer, list(t.names)),
                "owner": t.owner,
                "stubs": [worker_service_pb2_grpc.WorkerServiceStub(self.router.channel(a))
                          for a in t.peers],
            }
            for t in tied
        ]

    def tied_grad(self, request, context):
        """Owner side: a replica's gradient for a tied parameter."""
        if request.init_token == self._tied_token:
            grad = deserialize_tensor(request.tensor)
            with self._tied_cond:
                self._tied_grads.setdefault((request.group, request.batch_id), []).append(grad)
                self._tied_cond.notify_all()
        return worker_service_pb2.Ack(batch_id=request.batch_id)

    def tied_update(self, request, context):
        """Replica side: the owner's value after its optimizer step."""
        if request.init_token == self._tied_token:
            value = deserialize_tensor(request.tensor)
            for t in self._tied:
                if t["group"] == request.group:
                    with torch.no_grad():
                        t["param"].copy_(value)
        return worker_service_pb2.Ack(batch_id=request.batch_id)

    def _tied_call(self, method, stub, group: int, key: int, tensor: torch.Tensor) -> None:
        try:
            getattr(stub, method)(worker_service_pb2.TiedTensor(
                group=group, batch_id=key, init_token=self._tied_token,
                tensor=serialize_tensor(tensor)))
        except Exception as exc:
            if self._is_unavailable_rpc(exc):
                raise _PeerUnavailableError(
                    f"peer unavailable phase={method} group={group} batch_id={key}") from exc
            raise

    def _optimize(self, batch_id: int) -> None:
        """Optimizer step with tied-parameter sync across workers.

        Replicas step first (they are downstream), so they hand their gradient to the
        owner and drop it; the owner adds every replica gradient, steps, and pushes the
        new value back before this batch completes — the next batch sees it everywhere.
        """
        synced = [t for t in self._tied if t["stubs"] and t["param"].requires_grad]
        key = batch_id // self._n_micro if self._n_micro > 1 else batch_id
        for t in synced:
            if not t["owner"]:
                p = t["param"]
                self._tied_call("tied_grad", t["stubs"][0], t["group"], key,
                                p.grad if p.grad is not None else torch.zeros_like(p))
                p.grad = None
        for t in synced:
            if t["owner"]:
                k = (t["group"], key)
                with self._tied_cond:
                    if not self._tied_cond.wait_for(
                            lambda: len(self._tied_grads.get(k, ())) >= len(t["stubs"]),
                            timeout=_TIED_GRAD_TIMEOUT_S):
                        raise RuntimeError(f"tied gradient missing group={t['group']} batch_id={key}")
                    grads = self._tied_grads.pop(k)
                p = t["param"]
                total = sum(g.to(p.device) for g in grads)
                p.grad = total if p.grad is None else p.grad + total
        self.layer.optimize()
        for t in synced:
            if t["owner"]:
                for stub in t["stubs"]:
                    self._tied_call("tied_update", stub, t["group"], key, t["param"].detach())

    # ── NAT ────────────────────────────────────────────────────────────────────

    def _open_peer_tunnels(self, request) -> None:
        """A NAT'd worker can't be dialed, so it opens a tunnel to every node that sends
        to it: its neighbours (activations, gradients) and tied-weight peers. The
        coordinator's tunnel is already open. Through a relay nothing is needed."""
        if self.router.relay:
            return
        wanted = {a for a in (request.prev_worker, request.next_worker, request.label_worker,
                              *request.prev_workers) if a}
        wanted |= {a for t in request.tied for a in t.peers}
        wanted -= {request.coordinator, *self.router.addresses()}
        for address in sorted(self._peer_tunnels - wanted):
            self.router.release(address)
        for address in sorted(wanted):
            try:
                self.router.keep(address, wait=30.0)
            except grpc.RpcError as exc:
                raise RuntimeError(
                    f"this worker is behind NAT (transport.nat) and can't open a tunnel to its "
                    f"peer {address}: {exc.details()}. If that peer is behind NAT too, route "
                    "both through a relay (transport.relay).") from None
        self._peer_tunnels = wanted

    # ── capabilities (planning) ────────────────────────────────────────────────

    def capabilities(self) -> dict:
        """What this worker tells the planner (fields of ``NodeInfo``)."""
        gflops = self.declared["compute_gflops"]
        if not gflops:
            if self._measured_gflops is None:
                self._measured_gflops = measure_compute_gflops(self.device)
            gflops = self._measured_gflops
        dev = str(self.device)
        return {
            "device":         dev,
            "memory_mb":      get_available_memory_mb(dev),
            "compute_gflops": float(gflops),
            "max_memory_mb":  int(self.declared["max_memory_mb"]),
            "first_slice":    bool(self.declared["first_slice"]),
            "tags":           list(self.declared["tags"]),
        }

    def describe(self, request, context):
        return worker_service_pb2.NodeDescription(hostname=_HOSTNAME, **self.capabilities())

    def echo(self, request, context):
        return worker_service_pb2.EchoReply(n_bytes=len(request.payload))

    def probe_link(self, request, context):
        try:
            stub = worker_service_pb2_grpc.WorkerServiceStub(self.router.channel(request.target))
            rtt, mbps = measure_link(stub, request.n_bytes, max(1, request.repeats or 3))
            return worker_service_pb2.LinkProbeReply(rtt_ms=rtt, bandwidth_mbps=mbps)
        except Exception as e:
            return worker_service_pb2.LinkProbeReply(error=str(e))

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
        if epoch < 0:
            # liveness probe (heartbeats, failure-report checks): read nothing, reset nothing
            return worker_service_pb2.WorkerStatsResponse(
                worker_index = self._worker_index,
                run_id       = self._run_id,
                epoch        = epoch,
                fatal_error  = (self._fatal_error or {}).get("message", ""),
            )
        summary = self._profiler.epoch_summary(epoch)
        records = self._profiler.batch_records()
        self._profiler.reset_epoch()
        comm, compute_s = self._comm, self._compute_s
        self._reset_comm()

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
            fatal_error  = (self._fatal_error or {}).get("message", ""),
            compute_ms   = compute_s * 1e3,
            **comm,
        )

    # ── client nodes (sl.client_nodes, M10) ─────────────────────────────────────

    def client_info(self, request, context):
        """Its loader's batches and samples per epoch, for the coordinator's schedule."""
        loader = self._client_data()
        try:
            n_samples = len(loader.dataset)
        except (AttributeError, TypeError):
            n_samples = len(loader)
        return worker_service_pb2.ClientInfo(n_batches=len(loader), n_samples=n_samples)

    def client_step(self, request, context):
        """Take the next batch of this client's loader and run it as ``batch_id``: labels
        to the last slice, inputs into this slice (the coordinator never sees the data)."""
        loader = self._client_data()
        with self._data_lock:
            if request.epoch != self._data_epoch or request.index == 0:   # a new or retried epoch
                self._start_epoch(loader, request.epoch)
            if request.index != self._data_index:
                raise RuntimeError(f"client_step: batch {request.index} of epoch {request.epoch} "
                                   f"asked, but this client is at batch {self._data_index}")
            inputs, labels = self._next_batch(loader)
        self._drive_batch(request.batch_id, inputs, labels)
        return worker_service_pb2.Ack(batch_id=request.batch_id)

    def client_round(self, request, context):
        """``sl.client_schedule: concurrent``: run this round's batches of the loader (from
        its start) as ``batch_ids``, one in flight: each batch's labels and input go out,
        then the next waits for this slice's backward of the previous. Returns when the last
        is done; a fatal error here, or a re-init / shutdown (new generation), ends it."""
        loader = self._client_data()
        generation = self._generation
        with self._data_lock:
            self._start_epoch(loader, request.epoch)
            for batch_id in request.batch_ids:
                inputs, labels = self._next_batch(loader)
                done = threading.Event()
                with self._lock:
                    self._round_done[batch_id] = done
                self._drive_batch(batch_id, inputs, labels)
                while not done.wait(timeout=1.0):
                    if self._fatal_error is not None:
                        raise RuntimeError(f"client_round: {self._fatal_error['message']}")
                    if self._generation != generation:
                        raise RuntimeError("client_round: the slice was re-initialized or shut down")
        return worker_service_pb2.Ack(batch_id=request.batch_ids[-1] if request.batch_ids else 0)

    def _start_epoch(self, loader, epoch: int) -> None:
        """Iterate ``loader`` from its start for ``epoch``. A new epoch remembers the random
        state it starts from (the loader's shuffle, dropout); the same epoch again (retried
        after a server worker's recovery) restores it, so the replay takes the same batches."""
        from ..core.determinism import rng_restore, rng_snapshot
        if epoch != self._data_epoch or self._data_rng is None:
            self._data_rng = rng_snapshot(loader)
        else:
            rng_restore(self._data_rng, loader)
        self._data_epoch, self._data_iter, self._data_index = epoch, iter(loader), 0

    def _next_batch(self, loader):
        """The epoch's next batch. After the last one, run the iterator to its end, as a for
        loop does: a shuffling ``RandomSampler`` draws its generator once more there, so
        without it every later epoch's order differs from the in-process session's."""
        batch = next(self._data_iter)
        self._data_index += 1
        try:
            last = self._data_index == len(loader)
        except TypeError:                 # no length (an iterable dataset)
            last = False
        if last:
            for _ in self._data_iter:
                pass
        return batch

    def _client_data(self):
        if self.data is None:
            raise RuntimeError("this worker is not a client node: it has no data (start it "
                               "with data=, e.g. ts.run ROLE=client)")
        return self.data

    def _drive_batch(self, batch_id: int, inputs, labels: torch.Tensor) -> None:
        """What the coordinator does for a batch (``DistributedExecutor._send_batch``),
        done by the client that owns it: per (micro-)batch, the labels to the last slice
        and the input into this slice. The label call runs while this slice computes (the
        last slice takes the two in either order): its round trip was on every batch's
        critical path. Its result is still awaited, so an error fails the step."""
        from .distributed import _unpack_inputs
        main, aux = _unpack_inputs(inputs)
        M = self._n_micro
        if M == 1:
            parts = [(batch_id, main, labels, aux)]
        else:
            mains, ys = main.chunk(M), labels.chunk(M)
            parts = [(batch_id * M + m, mains[m], ys[m],
                      {k: v.chunk(M)[m] if v.dim() > 0 and v.shape[0] == main.shape[0] else v
                       for k, v in aux.items()}) for m in range(M)]
        sent = []
        for mbid, x, y, a in parts:
            self._comm["bytes_label"] += tensor_nbytes(y)
            self._comm["msgs_label"] += 1
            req = worker_service_pb2.ForwardRequest(batch_id=mbid, label=serialize_tensor(y))
            sent.append(self._label_calls.submit(contextvars.copy_context().run,
                                                 self._label_stub.forward, req))
            self.submit_input(mbid, x, a)
        for f in sent:
            f.result()

    def _read_state(self, weights_only: bool = False) -> dict:
        """This slice's weights and optimizer state (none with ``weights_only``), as CPU copies."""
        from ..sl.clients import _copy_optimizer_state
        opt = None if weights_only else self.layer.optimizer
        return {"model": {k: v.detach().to("cpu", copy=True) for k, v in self.layer.state_dict().items()},
                "optimizer": _copy_optimizer_state(opt.state_dict(), "cpu") if opt else None}

    def _write_state(self, state: dict) -> None:
        """Load a state (copying it: the caller may reuse it). No optimizer state means
        a fresh one, as in-process (``SplitLayerClientState.set_state``)."""
        from ..sl.clients import _copy_optimizer_state
        self.layer.load_state_dict(state["model"])
        opt = self.layer.optimizer
        if opt is None:
            return
        if state.get("optimizer") is not None:
            opt.load_state_dict(_copy_optimizer_state(state["optimizer"], self.device))
        else:
            opt.state.clear()

    def _select_client(self, batch_id: int) -> None:
        """SplitFed V1 (``per_client_state``): put the copy of this slice that belongs to the
        batch's client in, keeping the previous client's. A client's first batch of a round
        starts from the slice as the round began. Runs on the compute pool, before the
        batch's forward; the coordinator keeps one batch in flight."""
        if not self._per_client:
            return
        k = (batch_id // self._n_micro) % self._n_clients
        if k == self._active_client:
            return
        if self._active_client is None:
            self._round_initial = self._read_state()
        else:
            self._client_copies[self._active_client] = self._read_state()
            self._write_state(self._client_copies.get(k, self._round_initial))
        self._active_client = k

    def get_state(self, request, context):
        """This slice's weights and optimizer state (CPU copies), e.g. for a handoff or an
        aggregation; with ``per_client``, its copy for ``client`` (SplitFed V1). Runs on the
        compute pool, so after any batch in progress."""

        def _read():
            if request.per_client and request.client != self._active_client:
                state = self._client_copies.get(request.client, self._round_initial)
                if state is None:
                    raise RuntimeError(f"no copy of this slice for client {request.client}")
            else:
                state = self._read_state(weights_only=request.weights_only)
            if request.weights_only:
                state = {"model": state["model"], "optimizer": None}
            buf = io.BytesIO()
            torch.save(state, buf)
            return buf.getvalue()

        return worker_service_pb2.StateBlob(data=self._pool.submit(_read).result())

    def set_state(self, request, context):
        """Load a state from :meth:`get_state` (``weights_only``: plain tensors and
        containers, no code); with ``reset_clients``, then drop the per-client copies, so
        the next round starts every client from this state (SplitFed V1)."""
        state = torch.load(io.BytesIO(request.data), weights_only=True) if request.data else None

        def _write():
            if state is not None:
                self._write_state(state)
            if request.reset_clients:
                self._client_copies.clear()
                self._active_client = self._round_initial = None

        self._pool.submit(_write).result()
        return worker_service_pb2.Ack(batch_id=0)

    # ── forward / backward RPCs (fire-and-forget) ──────────────────────────────

    def forward(self, request, context):
        generation = self._generation
        if request.label.data:
            self._pool.submit(_tracer.propagate_to_thread(self._forward), request, generation)
        else:
            self._submit_forward(request.batch_id, self._forward, request, generation)
        return worker_service_pb2.Ack(batch_id=request.batch_id)

    def _gated(self) -> bool:
        """Slice 1 of a server of several slices, serving client nodes: with clients
        running at once (``sl.client_schedule: concurrent``) its batches would interleave
        forward and backward, and a step would change weights a pending backward still
        needs. So it takes one batch at a time (all its micro-batches), end to end."""
        return bool(self._prev_peers) and not self.is_last

    def _submit_forward(self, batch_id: int, fn, *args) -> None:
        """Queue a forward on the compute pool; on a gated slice, park it while another
        batch is in flight (never block the pool: that batch's backward runs there)."""
        if self._gated():
            b = batch_id // self._n_micro
            with self._lock:
                if self._gate_batch is not None and self._gate_batch != b:
                    self._gate_parked.append((batch_id, fn, args))
                    return
                self._gate_batch = b
        self._pool.submit(_tracer.propagate_to_thread(fn), *args)

    def _gate_release(self, batch_id: int) -> None:
        """The gated slice finished ``batch_id``'s backward: admit the next parked batch."""
        if not self._gated():
            return
        with self._lock:
            if self._gate_batch != batch_id // self._n_micro:
                return
            self._gate_batch, admit = None, []
            if self._gate_parked:
                nxt = self._gate_parked[0][0] // self._n_micro
                admit = [p for p in self._gate_parked if p[0] // self._n_micro == nxt]
                self._gate_parked = [p for p in self._gate_parked if p[0] // self._n_micro != nxt]
                self._gate_batch = nxt
        for _, fn, args in admit:
            self._pool.submit(_tracer.propagate_to_thread(fn), *args)

    def backward(self, request, context):
        generation = self._generation
        self._pool.submit(_tracer.propagate_to_thread(self._backward), request, generation)
        return worker_service_pb2.Ack(batch_id=request.batch_id)

    def _take_label(self, batch_id: int, label: torch.Tensor, generation: int) -> None:
        """A label for the last slice: run the loss now if the activation is already
        here, else keep the label until it arrives."""
        with self._lock:
            self._ensure_generation(generation, "forward_last", batch_id)
            cached = self._outputs.pop(batch_id, None)
            if cached is None:
                self._labels[batch_id] = label
        if cached is not None:
            out, x_ref = cached
            self._run_backward_last(batch_id, out, x_ref, label, generation)

    def submit_label(self, batch_id: int, label: torch.Tensor) -> None:
        """In-process counterpart of ``forward(ForwardRequest(label=...))`` for the last
        slice: a U-shaped data owner keeps its labels without serializing them."""
        generation = self._generation

        def _run():
            try:
                self._ensure_generation(generation, "forward_last", batch_id)
                self._take_label(batch_id, label.to(self.device), generation)
            except _StaleWorkError as e:
                _LOG.info("%s", e)
            except Exception as e:
                self._report_fatal_error("label", batch_id, e)

        self._pool.submit(_tracer.propagate_to_thread(_run))

    def submit_input(self, batch_id: int, tensor: torch.Tensor, aux: dict | None = None) -> None:
        """In-process counterpart of ``forward(ForwardRequest(input=...))`` for the first
        slice: the P2P driver feeds its own slice without serializing the batch."""
        generation = self._generation

        def _run():
            try:
                self._ensure_generation(generation, "forward", batch_id)
                self._before_runtime_phase("forward", batch_id)
                self._profiler.begin_batch(batch_id)
                self._profiler.mark_idle_end("fwd")
                self._run_forward_stage(batch_id, tensor.to(self.device), generation,
                                        aux={k: v.to(self.device) for k, v in (aux or {}).items()})
            except _StaleWorkError as e:
                _LOG.info("%s", e)
            except _PeerUnavailableError as e:
                _LOG.warning("%s", e)
            except Exception as e:
                self._report_fatal_error("forward", batch_id, e)

        self._pool.submit(_tracer.propagate_to_thread(_run))

    # ── cuts: hooks, byte counting, compute time ──────────────────────────────

    def _clock(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        import time
        return time.perf_counter()

    def _ctx(self, cut: int, direction: str, batch_id: int) -> CutContext:
        return CutContext(cut, direction, batch_id, self._n_slices)

    def _encode_out(self, batch_id: int, out: torch.Tensor) -> torch.Tensor:
        """The tensor sent to the next slice (output cut = this slice's index)."""
        raw = tensor_nbytes(out)
        hook = self._hooks.get(self._worker_index)
        wire = out if hook is None else pack_payload(
            hook.encode(out, self._ctx(self._worker_index, "forward", batch_id)))
        self._comm["bytes_fwd"] += tensor_nbytes(wire)
        self._comm["raw_bytes_fwd"] += raw
        self._comm["msgs_fwd"] += 1
        return wire

    def _encode_grad(self, batch_id: int, grad: torch.Tensor) -> torch.Tensor:
        """The tensor sent to the previous slice (input cut = index - 1)."""
        cut = self._worker_index - 1
        raw = tensor_nbytes(grad)
        hook = self._hooks.get(cut)
        wire = grad if hook is None else pack_payload(
            hook.encode_grad(grad, self._ctx(cut, "backward", batch_id)))
        self._comm["bytes_bwd"] += tensor_nbytes(wire)
        self._comm["raw_bytes_bwd"] += raw
        self._comm["msgs_bwd"] += 1
        return wire

    def _decode_in(self, batch_id: int, tensor_cpu: torch.Tensor) -> torch.Tensor:
        """An activation from the previous slice, on this device (hooks decoded)."""
        cut = self._worker_index - 1
        hook = self._hooks.get(cut) if cut >= 0 else None
        if hook is None:
            return tensor_cpu.to(self.device)
        return hook.decode(unpack_payload(tensor_cpu, self.device),
                           self._ctx(cut, "forward", batch_id))

    def _decode_grad(self, batch_id: int, tensor_cpu: torch.Tensor) -> torch.Tensor:
        """A gradient from the next slice, on this device (hooks decoded)."""
        hook = self._hooks.get(self._worker_index)
        if hook is None:
            return tensor_cpu.to(self.device)
        return hook.decode_grad(unpack_payload(tensor_cpu, self.device),
                                self._ctx(self._worker_index, "backward", batch_id))

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
            self._select_client(batch_id)
            self._profiler.begin_batch(batch_id)
            self._profiler.mark_idle_end("fwd")

            if self.is_last:
                self._forward_last(batch_id, request, generation)
                return

            # NCCL path: shape/dtype arrive via gRPC (empty data = sentinel); tensor via dist.recv.
            if self._tensor_transport == "nccl" and self._nccl_future is not None:
                self._wait_nccl()
            if (self._tensor_transport == "nccl" and self._dist_group is not None
                    and not request.input.data):
                import torch.distributed as dist
                dtype = _DTYPE_MAP.get(request.input.dtype, torch.float32)
                shape = list(request.input.shape)
                tensor = torch.empty(shape, dtype=dtype, device=self.device)
                dist.recv(tensor, src=self._worker_index - 1, group=self._dist_group)
                tensor = self._decode_in(batch_id, tensor)
                self._run_forward_stage(batch_id, tensor, generation, aux={})
                return

            with _tracer.span(
                "torchslicer.tensor.deserialize",
                batch_id=batch_id,
                worker=_HOSTNAME,
                source="input",
            ):
                tensor_cpu = deserialize_tensor(request.input, self.device)
            with _tracer.span(
                "torchslicer.tensor.h2d",
                batch_id=batch_id,
                worker=_HOSTNAME,
                source="input",
                device=str(self.device),
            ):
                tensor = self._decode_in(batch_id, tensor_cpu)
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
                        aux_cpu = deserialize_tensor(v, self.device)
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
            self._select_client(batch_id)
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
                tensor = self._decode_in(batch_id, tensor_cpu)
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
            t0 = self._clock()
            with self._profiler.phase("forward"):
                out = self.layer(tensor, **aux) if aux else self.layer(tensor)
                x_ref = self.layer.x
            self._compute_s += self._clock() - t0
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
        out = self._encode_out(batch_id, out)
        try:
            # NCCL path: shape/dtype via gRPC (no data), tensor via dist.isend (_nccl_send).
            if self._tensor_transport == "nccl":
                if self._nccl_future is not None:
                    self._wait_nccl()
            if (self._tensor_transport == "nccl" and self._dist_group is not None):
                with _tracer.span(
                    "torchslicer.worker.activation_send",
                    batch_id=batch_id,
                    worker=_HOSTNAME,
                    peer=self.next_worker,
                    transport="nccl",
                ):
                    self._nccl_send(out, self._worker_index + 1, lambda: self._next_stub.forward(
                        worker_service_pb2.ForwardRequest(
                            batch_id=batch_id,
                            input=worker_service_pb2.Tensor(
                                shape=list(out.shape),
                                dtype=_TORCH_TO_DTYPE.get(out.dtype, worker_service_pb2.FLOAT32),
                            ),
                        )))
                return
            if (self._tensor_transport in {"tcp", "ipc"} and self._next_tensor is not None
                    and self.router.is_direct(self.next_worker)):
                with _tracer.span(
                    "torchslicer.worker.activation_send",
                    batch_id=batch_id,
                    worker=_HOSTNAME,
                    peer=self.next_worker,
                    transport=self._tensor_transport,
                ):
                    self._next_tensor.send_tensor(b"F", batch_id, out)
                return
            with _tracer.span(
                "torchslicer.worker.activation_send",
                batch_id=batch_id,
                worker=_HOSTNAME,
                peer=self.next_worker,
                transport="grpc",
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
            # NCCL: shape/dtype in input, no data → receive tensor from previous worker.
            if self._tensor_transport == "nccl" and self._nccl_future is not None:
                self._wait_nccl()
            is_nccl_input = (
                not is_label
                and not bool(request.input.data)
                and self._tensor_transport == "nccl"
                and self._dist_group is not None
            )

            if is_nccl_input:
                import torch.distributed as dist
                dtype = _DTYPE_MAP.get(request.input.dtype, torch.float32)
                shape = list(request.input.shape)
                tensor = torch.empty(shape, dtype=dtype, device=self.device)
                dist.recv(tensor, src=self._worker_index - 1, group=self._dist_group)
                tensor = self._decode_in(batch_id, tensor)
                with _tracer.span("torchslicer.worker.forward", batch_id=batch_id,
                                  worker=_HOSTNAME, is_last=True,
                                  input_shape=str(tuple(tensor.shape))):
                    t0 = self._clock()
                    with self._profiler.phase("forward"):
                        out   = self.layer(tensor)
                        x_ref = self.layer.x
                    self._compute_s += self._clock() - t0
                with self._lock:
                    self._ensure_generation(generation, "forward_last", batch_id)
                    label = self._labels.pop(batch_id, None)
                    if label is None:
                        self._outputs[batch_id] = (out, x_ref)
                if label is not None:
                    self._run_backward_last(batch_id, out, x_ref, label, generation)
                return

            if is_label:
                with _tracer.span(
                    "torchslicer.tensor.deserialize",
                    batch_id=batch_id,
                    worker=_HOSTNAME,
                    source="label",
                ):
                    label_cpu = deserialize_tensor(request.label, self.device)
                with _tracer.span(
                    "torchslicer.tensor.h2d",
                    batch_id=batch_id,
                    worker=_HOSTNAME,
                    source="label",
                    device=str(self.device),
                ):
                    label = label_cpu.to(self.device)
                self._take_label(batch_id, label, generation)
            else:
                with _tracer.span(
                    "torchslicer.tensor.deserialize",
                    batch_id=batch_id,
                    worker=_HOSTNAME,
                    source="input_last",
                ):
                    tensor_cpu = deserialize_tensor(request.input, self.device)
                with _tracer.span(
                    "torchslicer.tensor.h2d",
                    batch_id=batch_id,
                    worker=_HOSTNAME,
                    source="input_last",
                    device=str(self.device),
                ):
                    tensor = self._decode_in(batch_id, tensor_cpu)
                with _tracer.span(
                    "torchslicer.worker.forward",
                    batch_id=batch_id,
                    worker=_HOSTNAME,
                    is_last=True,
                    input_shape=str(tuple(tensor.shape)),
                ):
                    t0 = self._clock()
                    with self._profiler.phase("forward"):
                        out   = self.layer(tensor)
                        x_ref = self.layer.x
                    self._compute_s += self._clock() - t0

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

            # NCCL path: gradient shape/dtype in message, no data → receive via dist.recv.
            if self._tensor_transport == "nccl" and self._nccl_future is not None:
                self._wait_nccl()
            if (self._tensor_transport == "nccl" and self._dist_group is not None
                    and not request.gradient.data):
                import torch.distributed as dist
                dtype = _DTYPE_MAP.get(request.gradient.dtype, torch.float32)
                shape = list(request.gradient.shape)
                grad_in = torch.empty(shape, dtype=dtype, device=self.device)
                dist.recv(grad_in, src=self._worker_index + 1, group=self._dist_group)
                grad_in = self._decode_grad(batch_id, grad_in)
                self._run_backward_stage(batch_id, grad_in, is_last_micro, generation)
                return

            with _tracer.span(
                "torchslicer.tensor.deserialize",
                batch_id=batch_id,
                worker=_HOSTNAME,
                source="gradient",
            ):
                grad_cpu = deserialize_tensor(request.gradient, self.device)
            with _tracer.span(
                "torchslicer.tensor.h2d",
                batch_id=batch_id,
                worker=_HOSTNAME,
                source="gradient",
                device=str(self.device),
            ):
                grad_in = self._decode_grad(batch_id, grad_cpu)
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
                grad_in = self._decode_grad(batch_id, grad_cpu)
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
        if grad_in.dtype != out.dtype:
            grad_in = grad_in.to(out.dtype)

        t0 = self._clock()
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
                elif out.requires_grad:   # else nothing here needs a gradient (SplitLayer.backward)
                    out.backward(grad_in)
                grad = x_ref.grad if x_ref is not None else None
                if grad is None and x_ref is not None and x_ref.requires_grad:
                    grad = torch.zeros_like(x_ref)    # the input reached nothing differentiable
            if is_last_micro:
                with self._profiler.phase("optimizer"):
                    self._optimize(batch_id)
        self._compute_s += self._clock() - t0

        del out
        try:
            self._send_backward(batch_id, grad, is_last_micro, generation)
        finally:
            if is_last_micro:
                self._gate_release(batch_id)

    def _run_backward_last(self, batch_id: int, out: torch.Tensor,
                           x_ref: torch.Tensor, label: torch.Tensor, generation: int):
        try:
            self._ensure_generation(generation, "backward_last", batch_id)
            self._before_runtime_phase("backward_last", batch_id)
            n_micro       = self._n_micro
            is_last_micro = (n_micro <= 1) or (batch_id % n_micro == n_micro - 1)
            full_batch_id = batch_id // n_micro if n_micro > 1 else batch_id

            t0 = self._clock()
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
                        self._optimize(batch_id)
            self._compute_s += self._clock() - t0

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
        prev_worker, prev_stub, prev_tensor = self._prev_for(batch_id)
        if prev_stub or prev_tensor:
            grad = self._encode_grad(batch_id, grad)
            with self._profiler.phase("send_bwd"):
                try:
                    # NCCL path: send metadata via gRPC, tensor via dist.isend.
                    if self._tensor_transport == "nccl" and self._nccl_future is not None:
                        self._wait_nccl()
                    if (self._tensor_transport == "nccl" and self._dist_group is not None):
                        with _tracer.span(
                            "torchslicer.worker.gradient_send",
                            batch_id=batch_id,
                            worker=_HOSTNAME,
                            peer=prev_worker,
                            transport="nccl",
                        ):
                            self._nccl_send(grad, self._worker_index - 1, lambda: prev_stub.backward(
                                worker_service_pb2.BackwardRequest(
                                    batch_id=batch_id,
                                    gradient=worker_service_pb2.Tensor(
                                        shape=list(grad.shape),
                                        dtype=_TORCH_TO_DTYPE.get(grad.dtype, worker_service_pb2.FLOAT32),
                                    ),
                                )))
                    elif (self._tensor_transport in {"tcp", "ipc"} and prev_tensor is not None
                            and self.router.is_direct(prev_worker)):
                        with _tracer.span(
                            "torchslicer.worker.gradient_send",
                            batch_id=batch_id,
                            worker=_HOSTNAME,
                            peer=prev_worker,
                            transport=self._tensor_transport,
                        ):
                            prev_tensor.send_tensor(b"B", batch_id, grad)
                    else:
                        with _tracer.span(
                            "torchslicer.worker.gradient_send",
                            batch_id=batch_id,
                            worker=_HOSTNAME,
                            peer=prev_worker,
                            transport="grpc",
                        ):
                            prev_stub.backward(worker_service_pb2.BackwardRequest(
                                batch_id=batch_id,
                                gradient=serialize_tensor(grad),
                            ))
                except Exception as exc:
                    if isinstance(exc, _TensorStreamError) or self._is_unavailable_rpc(exc):
                        raise _PeerUnavailableError(
                            f"peer unavailable phase=backward batch_id={batch_id} peer={prev_worker}"
                        ) from exc
                    raise
            self._profiler.mark_idle_start("fwd")
            self._profiler.end_batch()
        elif is_last_micro:
            self._coord_stub.batch_done(coordinator_service_pb2.BatchDoneRequest(
                batch_id=batch_id,
                run_id=self._run_id,
            ))
            with self._lock:
                done = self._round_done.pop(batch_id // self._n_micro, None)
            if done is not None:
                done.set()
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


# ── start_worker / run_worker ──────────────────────────────────────────────────

class WorkerHandle:
    """A worker started in this process by :func:`start_worker`."""

    def __init__(self, servicer, server, address: str):
        self.servicer = servicer
        self.server   = server
        self.address  = address

    def stop(self, grace: float = 0) -> None:
        self.servicer.stop_tensor_server()
        self.server.stop(grace)
        self.servicer.router.close()


class _NoServer:
    """What a NAT'd worker has instead of a gRPC server: it listens on nothing and is
    served through its tunnels. ``stop`` ends it like ``grpc.Server.stop``."""

    def __init__(self, router):
        self._router = router
        self._stopped = threading.Event()

    def stop(self, grace=None):
        # like grpc.Server.stop(grace): let in-flight replies (e.g. the Ack of the
        # shutdown call that stops it) leave over the tunnel before closing it
        if grace:
            import time
            time.sleep(min(float(grace), 1.0))
        self._stopped.set()
        self._router.close()

    def wait_for_termination(self, timeout=None):
        return self._stopped.wait(timeout)


def start_worker(port: int = 0, *, host: str = "[::]", address: str | None = None,
                 run_config: RunConfig | None = None, servicer_class=None,
                 grpc_workers: int = 10, nat: bool | None = None,
                 tunnel_to: list | None = None, data=None,
                 client_index: int | None = None) -> WorkerHandle:
    """Start a worker's gRPC (and, with ``transport.tensor: tcp``, TCP tensor) server in
    this process and return without blocking. It registers with no one: a coordinator
    or P2P driver must know its address (``StaticDiscovery``). ``run_worker`` is the
    blocking entry point built on it.

    ``port=0`` picks a free port (gRPC only; TCP needs a fixed ``port + offset``).
    Handy for local simulations, tests and the tutorials::

        followers = [ts.start_worker(host="127.0.0.1") for _ in range(2)]
        peers = [w.address for w in followers]

    ``nat`` (default ``transport.nat``): the worker is behind NAT. It opens no port;
    ``address`` is only its name, and it opens a tunnel to each node in ``tunnel_to``
    (the coordinator or P2P driver that will drive it), or to ``transport.relay``.

    ``data`` (a loader of ``(inputs, labels)``) makes it a client node (``sl.client_nodes``):
    it holds slice 0 and its own data, and declares the tags ``client`` and
    ``client=<client_index>``, by which the coordinator orders the clients.
    """
    cfg = run_config or RunConfig.load()
    nat = bool(cfg.transport.nat if nat is None else nat)
    relay = (cfg.transport.relay or "").strip()
    servicer = (servicer_class or WorkerServicer)()
    if cfg.network.device and cfg.network.device != "auto":
        # the servicer resolved DEVICE from the env; an explicit config device wins
        servicer.device = torch.device(resolve_device(cfg.network.device))
    servicer._tensor_transport   = cfg.transport.tensor.strip().lower()
    servicer._tensor_port_offset = int(cfg.transport.tensor_port_offset)
    servicer._persistent         = bool(cfg.network.worker_persistent)
    servicer.declared = {
        "compute_gflops": float(cfg.network.worker_compute_gflops),
        "max_memory_mb":  int(cfg.network.worker_max_memory_mb),
        "first_slice":    bool(cfg.network.worker_first_slice),
        "tags":           list(cfg.network.worker_tags) + _client_tags(data, client_index),
    }
    servicer.data = data
    servicer.router = router = Router(relay or None, keepalive_s=cfg.transport.tunnel_keepalive_s)
    servicer.nat = nat
    if nat:
        tunnel_to = [a for a in (tunnel_to or []) if a]
        if not relay and not tunnel_to:
            router.close()
            raise ValueError("a worker behind NAT (transport.nat) must know who drives it: "
                             "network.coordinator_address (the coordinator or P2P driver), "
                             "or transport.relay")
        if address is None:
            address = f"{_HOSTNAME}:{port}" if port else f"{_HOSTNAME}-{os.getpid()}-{id(servicer):x}:0"
        server = _NoServer(router)
        servicer.set_server(server)
        router.serve(address, servicer, worker_service_pb2_grpc.add_WorkerServiceServicer_to_server)
        if not relay:
            for target in tunnel_to:
                router.keep(target)
        _LOG.info("worker behind NAT address=%s tunnels=%s", address, [relay] if relay else tunnel_to)
        return WorkerHandle(servicer, server, address)
    if port == 0 and servicer._tensor_transport in ("tcp", "ipc"):
        raise ValueError("start_worker(port=0) needs transport.tensor grpc or nccl: the tcp/ipc "
                         "tensor port is port + offset")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=grpc_workers), options=_GRPC_OPTS)
    worker_service_pb2_grpc.add_WorkerServiceServicer_to_server(servicer, server)
    router.add_to_server(server)
    port = server.add_insecure_port(f"{host}:{port}")
    server.start()
    servicer.set_server(server)
    servicer.start_tensor_server(port + servicer._tensor_port_offset)
    if address is None:
        address = f"127.0.0.1:{port}" if host in ("127.0.0.1", "localhost") else f"{_HOSTNAME}:{port}"
    router.serve(address, servicer, worker_service_pb2_grpc.add_WorkerServiceServicer_to_server)
    return WorkerHandle(servicer, server, address)



def _client_tags(data, client_index) -> list:
    if data is None:
        return []
    if client_index is None:
        raise ValueError("a client node (data=) needs its client_index")
    return ["client", f"client={int(client_index)}"]


def run_worker(
    port=None,
    coordinator_addr=_UNSET,
    worker_address=None,
    device=None,
    servicer_class=None,
    grpc_workers=10,
    data=None,
    client_index=None,
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
        data, client_index: a client node's loader and index (``sl.client_nodes``):
                          see :func:`start_worker`.

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

    # ── resolve worker tags ─────────────────────────────────────────────────
    tags = list(cfg.network.worker_tags) + _client_tags(data, client_index)

    # ── build servicer + gRPC server ────────────────────────────────────────
    # Behind NAT, the worker tunnels to whoever drives it: the coordinator, else the
    # P2P driver (network.coordinator_address, else the first of discovery.peers).
    nat = bool(cfg.transport.nat)
    driver = coordinator_addr or cfg.network.coordinator_address or next(
        (p for p in cfg.discovery.peers if p != worker_address), None)
    handle   = start_worker(port, address=worker_address, run_config=cfg,
                            servicer_class=servicer_class, grpc_workers=grpc_workers,
                            tunnel_to=[driver] if nat else None, data=data,
                            client_index=client_index)
    servicer, server = handle.servicer, handle.server
    tag_str = f"  tags=[{', '.join(tags)}]" if tags else ""
    _LOG.info("worker started port=%s hostname=%s device=%s%s", port, _HOSTNAME, resolved_device, tag_str)

    # ── announce to coordinator (centralized topology only) ─────────────────
    if coordinator_addr:
        from ..discovery import NodeInfo, announce_to_coordinator
        # declared + measured capabilities (one short matmul benchmark, cached)
        caps = servicer.capabilities()
        node_info = NodeInfo(
            node_id=worker_address,  # unique per worker even when several share a host
            address=worker_address,
            **caps,
        )
        _LOG.info("capabilities %s", caps)
        _LOG.info("registering with coordinator address=%s", coordinator_addr)
        policy = RetryPolicy(max_attempts=cfg.discovery.registration_max_attempts,
                             delay_s=cfg.discovery.registration_delay_s,
                             rpc_timeout_s=cfg.discovery.registration_rpc_timeout_s)
        channel = servicer.router.channel(coordinator_addr)
        if nat and not servicer.router.relay:
            # the coordinator calls this worker (init) as soon as it registers
            if not servicer.router.keep(coordinator_addr).wait(policy.max_attempts * policy.delay_s):
                _LOG.warning("no tunnel to the coordinator yet; registering anyway")
        try:
            result = announce_to_coordinator(
                coordinator_addr,
                node_info,
                channel=channel,
                retry_policy=RetryPolicy(
                    max_attempts=cfg.discovery.registration_max_attempts,
                    delay_s=cfg.discovery.registration_delay_s,
                    rpc_timeout_s=cfg.discovery.registration_rpc_timeout_s,
                ),
            )
            _LOG.info("registration complete run_id=%s worker_index=%s", result.run_id, result.worker_index)
        except RuntimeError as e:
            if not servicer._persistent:
                _LOG.error("registration failed fatally: %s", e)
                server.stop(0)
                sys.exit(1)
            # Persistent workers outlive coordinators: keep retrying via the watchdog.
            _LOG.warning("registration failed, worker persistent — retrying via watchdog: %s", e)

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
                        coordinator_addr, node_info, channel=channel,
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
