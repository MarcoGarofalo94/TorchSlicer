import io
import os
import sys
import time
import threading
import socket

import grpc
import torch
from torch import nn, optim
from concurrent import futures

_MAX_MSG = 256 * 1024 * 1024  # 256 MB
_GRPC_OPTS = [
    ('grpc.max_send_message_length',    _MAX_MSG),
    ('grpc.max_receive_message_length', _MAX_MSG),
]

def _channel(addr): return grpc.insecure_channel(addr, options=_GRPC_OPTS)

from torchslicer.transport.grpc.coordinator import coordinator_service_pb2_grpc, coordinator_service_pb2
from torchslicer.transport.grpc.worker import worker_service_pb2_grpc, worker_service_pb2

from torchslicer.core.split_layer import SplitLayer
from torchslicer.discovery import NodeInfo, announce_to_coordinator
from torchslicer.monitor import tracer as _tracer
from torchslicer.monitor import WorkerProfiler


# ── helpers ────────────────────────────────────────────────────────────────────

_DTYPE_MAP = {
    worker_service_pb2.FLOAT32:  torch.float32,
    worker_service_pb2.FLOAT16:  torch.float16,
    worker_service_pb2.BFLOAT16: torch.bfloat16,
    worker_service_pb2.FLOAT64:  torch.float64,
    worker_service_pb2.INT64:    torch.int64,
    worker_service_pb2.INT32:    torch.int32,
}
_TORCH_TO_DTYPE = {v: k for k, v in _DTYPE_MAP.items()}


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    arr = t.detach().cpu().contiguous()
    if arr.dtype == torch.bfloat16:
        return arr.view(torch.uint8).numpy().tobytes()
    return arr.numpy().tobytes()


def deserialize_tensor(msg: worker_service_pb2.Tensor) -> torch.Tensor:
    dtype = _DTYPE_MAP.get(msg.dtype, torch.float32)
    return torch.frombuffer(bytearray(msg.data), dtype=dtype).reshape(list(msg.shape)).clone()


def serialize_tensor(t: torch.Tensor) -> worker_service_pb2.Tensor:
    return worker_service_pb2.Tensor(
        data=_tensor_to_bytes(t),
        shape=list(t.shape),
        dtype=_TORCH_TO_DTYPE.get(t.dtype, worker_service_pb2.FLOAT32),
    )


def _get_available_memory_mb(device: str) -> int:
    """Return available memory in MB (GPU free, or system RAM approximation)."""
    try:
        if device.startswith("cuda"):
            dev_idx = int(device.split(":")[-1]) if ":" in device else 0
            free, _ = torch.cuda.mem_get_info(dev_idx)
            return free // (1024 * 1024)
        import psutil
        return psutil.virtual_memory().available // (1024 * 1024)
    except Exception:
        return 0


def _current_mem_mb(device) -> float:
    """Return currently allocated GPU memory in MB, or 0 for CPU."""
    try:
        if device.type == "cuda":
            return torch.cuda.memory_allocated(device) / (1024 * 1024)
    except Exception:
        pass
    return 0.0


# ── WorkerServicer ─────────────────────────────────────────────────────────────

class WorkerServicer(worker_service_pb2_grpc.WorkerServiceServicer):

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._server = None       # set by serve() after grpc.server() is created

        # Per-run state — reset fully on every init() call
        self._reset_state()

        # Single-worker compute pool: serialises all forward/backward ops on
        # this worker so concurrent micro-batch RPCs never race on self.layer.x
        # or the model's .grad tensors.  Pipeline parallelism still happens
        # because DIFFERENT workers run their compute pools concurrently.
        self._pool = futures.ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()

    def set_server(self, server):
        """Called by serve() so shutdown() can stop the gRPC server."""
        self._server = server

    def _reset_state(self):
        """Clear all per-run state. Called at the top of init() for clean re-use."""
        self.layer:       SplitLayer = None
        self.loss_fn                 = None
        self.is_last:     bool       = False
        self.prev_worker: str        = None
        self.next_worker: str        = None
        self.coordinator: str        = None
        self._n_micro:    int        = 1
        self._run_id:     str        = ""
        self._worker_index: int      = 0

        # Persistent stubs — rebuilt each init()
        self._next_stub = None
        self._prev_stub = None
        self._coord_stub = None

        # Per-batch state, keyed by batch_id
        self._labels:      dict = {}
        self._outputs:     dict = {}   # batch_id → (out, x_ref)
        self._micro_losses: dict = {}  # full_batch_id → accumulated loss

        # Profiler — re-created on init() with the run's profile settings
        self._profiler: WorkerProfiler = WorkerProfiler(verbosity=0)

    # ── init ───────────────────────────────────────────────────────────────────

    def init(self, request, context):
        try:
            # Full state reset so workers can be reused across runs
            self._reset_state()

            layers = self._build_layers(request.layers)
            self.layer   = SplitLayer(layers, is_last=request.is_last)
            self.is_last = request.is_last
            self._run_id       = request.run_id
            self._worker_index = request.worker_index

            if request.is_last and request.HasField("criterion"):
                crit_extra = (torch.load(io.BytesIO(request.criterion.extra_params), weights_only=False)
                              if request.criterion.extra_params else {})
                self.loss_fn = getattr(nn, request.criterion.name)(**crit_extra)

            opt_extra = (torch.load(io.BytesIO(request.optimizer.extra_params), weights_only=False)
                         if request.optimizer.extra_params else {})
            opt = getattr(optim, request.optimizer.name)(
                self.layer.parameters(), lr=request.optimizer.lr, **opt_extra)
            self.layer.set_optimizer(opt)

            self.prev_worker = request.prev_worker or None
            self.next_worker = request.next_worker or None
            self.coordinator = request.coordinator
            self._n_micro    = max(1, request.n_micro) if request.n_micro else 1

            # Build persistent stubs so forward/backward never open a channel
            if self.next_worker:
                self._next_stub = worker_service_pb2_grpc.WorkerServiceStub(
                    _channel(self.next_worker))
            if self.prev_worker:
                self._prev_stub = worker_service_pb2_grpc.WorkerServiceStub(
                    _channel(self.prev_worker))
            self._coord_stub = coordinator_service_pb2_grpc.CoordinatorServiceStub(
                _channel(self.coordinator))

            # Load checkpoint if resuming
            if request.checkpoint_path:
                self._load_checkpoint(request.checkpoint_path)

            self.layer = self.layer.to(self.device)
            if self.loss_fn:
                self.loss_fn = self.loss_fn.to(self.device)

            # Initialise profiler from profile settings in SliceConfig
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

            print(f"[init] run_id={self._run_id}  index={self._worker_index}  "
                  f"layers={layer_names}  is_last={self.is_last}  device={self.device}")
            print(f"       prev={self.prev_worker}  next={self.next_worker}")
            print(f"       param_mb={param_mb}  cuda_alloc_mb={cuda_alloc_mb}  n_micro={self._n_micro}")
            print(f"       profile_verbosity={request.profile_verbosity}  "
                  f"profile_memory={request.profile_memory}")

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
            print(f"[init] ERROR: {e}")
            import traceback; traceback.print_exc()
            return worker_service_pb2.StatusMessage(
                ok=False, message=str(e), hostname=socket.gethostname())

    # ── shutdown ───────────────────────────────────────────────────────────────

    def shutdown(self, request, context):
        """Save checkpoint if requested, then stop the gRPC server."""
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
                    print(f"[shutdown] checkpoint save failed: {e}")
                    import traceback; traceback.print_exc()

            print(f"[shutdown] worker {socket.gethostname()} stopping ...")
            if self._server:
                self._server.stop(grace=1)

        # Run in a thread so the RPC can return an Ack before we stop the server
        threading.Thread(target=_do_shutdown, daemon=True).start()
        return worker_service_pb2.Ack(batch_id=0)

    # ── get_stats RPC ──────────────────────────────────────────────────────────

    def get_stats(self, request, context):
        """Return epoch profiling stats and reset the profiler for the next epoch."""
        epoch   = request.epoch
        summary = self._profiler.epoch_summary(epoch)
        records = self._profiler.batch_records()
        self._profiler.reset_epoch()

        def _phase_stats(name: str) -> worker_service_pb2.PhaseStats:
            return worker_service_pb2.PhaseStats(
                avg_ms    = summary.get(f"{name}_avg_ms",   0.0),
                min_ms    = summary.get(f"{name}_min_ms",   0.0),
                max_ms    = summary.get(f"{name}_max_ms",   0.0),
                p95_ms    = summary.get(f"{name}_p95_ms",   0.0),
                total_ms  = summary.get(f"{name}_total_ms", 0.0),
                count     = summary.get("n_batches",        0),
                peak_mem_mb = summary.get(f"{name}_peak_mem_mb", 0.0),
            )

        batch_stats = [
            worker_service_pb2.BatchStats(
                batch_id    = r["batch_id"],
                forward_ms  = r["forward_ms"],
                backward_ms = r["backward_ms"],
                optimizer_ms= r["optimizer_ms"],
                send_fwd_ms = r["send_fwd_ms"],
                send_bwd_ms = r["send_bwd_ms"],
                idle_fwd_ms = r["idle_fwd_ms"],
                idle_bwd_ms = r["idle_bwd_ms"],
                peak_mem_mb = r["peak_mem_mb"],
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
            end_mem_mb   = round(_current_mem_mb(self.device), 2),
            n_batches    = summary.get("n_batches", 0),
            batches      = batch_stats,
        )

    # ── forward / backward RPCs (fire-and-forget) ──────────────────────────────

    def forward(self, request, context):
        self._pool.submit(self._forward, request)
        return worker_service_pb2.Ack(batch_id=request.batch_id)

    def backward(self, request, context):
        self._pool.submit(self._backward, request)
        return worker_service_pb2.Ack(batch_id=request.batch_id)

    # ── internal forward ───────────────────────────────────────────────────────

    def _forward(self, request: worker_service_pb2.ForwardRequest):
        try:
            batch_id = request.batch_id
            self._profiler.begin_batch(batch_id)
            self._profiler.mark_idle_end("fwd")

            if self.is_last:
                self._forward_last(batch_id, request)
                return

            tensor = deserialize_tensor(request.input).to(self.device)
            with _tracer.span(
                "torchslicer.worker.forward",
                batch_id=batch_id,
                worker=socket.gethostname(),
                input_shape=str(tuple(tensor.shape)),
            ) as s:
                with self._profiler.phase("forward"):
                    out   = self.layer(tensor)
                    x_ref = self.layer.x
                if s:
                    s.set_attribute("output_shape", str(tuple(out.shape)))

            with self._lock:
                self._outputs[batch_id] = (out, x_ref)

            with self._profiler.phase("send_fwd"):
                self._next_stub.forward(worker_service_pb2.ForwardRequest(
                    batch_id=batch_id,
                    input=serialize_tensor(out),
                ))

            # Worker is now idle waiting for gradient from the downstream chain
            self._profiler.mark_idle_start("bwd")

        except Exception as e:
            print(f"[forward] ERROR batch_id={request.batch_id}: {e}")
            import traceback; traceback.print_exc()

    def _forward_last(self, batch_id: int, request: worker_service_pb2.ForwardRequest):
        try:
            is_label = bool(request.label.data)

            if is_label:
                label = deserialize_tensor(request.label).to(self.device)
                with self._lock:
                    self._labels[batch_id] = label
                    cached = self._outputs.pop(batch_id, None)
                if cached is not None:
                    out, x_ref = cached
                    self._run_backward_last(batch_id, out, x_ref, label)
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
                    label = self._labels.pop(batch_id, None)
                    if label is None:
                        self._outputs[batch_id] = (out, x_ref)

                if label is not None:
                    self._run_backward_last(batch_id, out, x_ref, label)
        except Exception as e:
            print(f"[forward_last] ERROR batch_id={batch_id}: {e}")
            import traceback; traceback.print_exc()

    # ── internal backward ──────────────────────────────────────────────────────

    def _backward(self, request: worker_service_pb2.BackwardRequest):
        try:
            batch_id      = request.batch_id
            n_micro       = self._n_micro
            is_last_micro = (n_micro <= 1) or (batch_id % n_micro == n_micro - 1)

            self._profiler.mark_idle_end("bwd")

            grad_in = deserialize_tensor(request.gradient).to(self.device)

            with self._lock:
                cached = self._outputs.pop(batch_id, None)

            if cached is None:
                print(f"[backward] WARNING: no cached output for batch_id={batch_id}")
                return

            out, x_ref = cached

            with _tracer.span(
                "torchslicer.worker.backward",
                batch_id=batch_id,
                worker=socket.gethostname(),
                is_last=False,
            ):
                with self._profiler.phase("backward"):
                    out.backward(grad_in)
                    grad = x_ref.grad
                if is_last_micro:
                    with self._profiler.phase("optimizer"):
                        self.layer.optimize()

            del out
            if is_last_micro and self.device.type == "cuda":
                torch.cuda.empty_cache()
            self._send_backward(batch_id, grad, is_last_micro)
        except Exception as e:
            print(f"[backward] ERROR batch_id={request.batch_id}: {e}")
            import traceback; traceback.print_exc()

    def _run_backward_last(self, batch_id: int, out: torch.Tensor,
                           x_ref: torch.Tensor, label: torch.Tensor):
        try:
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
                    loss.backward()
                    grad = x_ref.grad
                if is_last_micro:
                    with self._profiler.phase("optimizer"):
                        self.layer.optimize()

            del out, label, loss, loss_unscaled
            if is_last_micro and self.device.type == "cuda":
                torch.cuda.empty_cache()
            self._send_backward(batch_id, grad, is_last_micro)
        except Exception as e:
            print(f"[backward_last] ERROR batch_id={batch_id}: {e}")
            import traceback; traceback.print_exc()

    def _send_backward(self, batch_id: int, grad: torch.Tensor, is_last_micro: bool = True):
        if self._prev_stub:
            with self._profiler.phase("send_bwd"):
                self._prev_stub.backward(worker_service_pb2.BackwardRequest(
                    batch_id=batch_id,
                    gradient=serialize_tensor(grad),
                ))
            # After sending gradient upstream, idle waiting for next forward
            self._profiler.mark_idle_start("fwd")
            self._profiler.end_batch()
        elif is_last_micro:
            self._coord_stub.batch_done(coordinator_service_pb2.BatchDoneRequest(
                batch_id=batch_id,
                run_id=self._run_id,
            ))
            # First worker: idle waiting for next forward from coordinator
            self._profiler.mark_idle_start("fwd")
            self._profiler.end_batch()

    # ── checkpoint ─────────────────────────────────────────────────────────────

    def _save_checkpoint(self, checkpoint_dir: str, run_id: str, epoch: int, worker_index: int):
        import os
        run_dir = os.path.join(checkpoint_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, f"worker_{worker_index}_epoch_{epoch}.pt")
        torch.save({
            "layer_state_dict":     self.layer.state_dict(),
            "optimizer_state_dict": self.layer.optimizer.state_dict() if self.layer.optimizer else None,
            "epoch":                epoch,
            "worker_index":         worker_index,
            "run_id":               run_id,
        }, path)
        print(f"[checkpoint] saved → {path}")

    def _load_checkpoint(self, checkpoint_path: str):
        import os
        if not os.path.exists(checkpoint_path):
            print(f"[resume] checkpoint not found: {checkpoint_path}, starting fresh")
            return
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.layer.load_state_dict(state["layer_state_dict"])
        if state.get("optimizer_state_dict") and self.layer.optimizer:
            self.layer.optimizer.load_state_dict(state["optimizer_state_dict"])
        print(f"[resume] loaded checkpoint: {checkpoint_path} "
              f"(epoch={state.get('epoch', '?')})")

    # ── helpers ────────────────────────────────────────────────────────────────

    def _build_layers(self, layer_configs) -> list:
        return [
            torch.load(io.BytesIO(lc.serialized), weights_only=False)
            for lc in layer_configs
        ]


# ── entrypoint ─────────────────────────────────────────────────────────────────

def serve():
    _tracer.auto_configure_if_env()

    port             = sys.argv[1] if len(sys.argv) > 1 else "50051"
    coordinator_addr = os.environ.get("COORDINATOR_ADDRESS", "coordinator:50054")
    hostname         = socket.gethostname()
    # WORKER_ADDRESS lets users override the address advertised to the coordinator
    # (useful when the worker's hostname is not routable from outside its container)
    node_address     = os.environ.get("WORKER_ADDRESS", f"{hostname}:{port}")

    # Start gRPC server before registering so the worker can handle RPCs
    servicer = WorkerServicer()
    server   = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=_GRPC_OPTS)
    worker_service_pb2_grpc.add_WorkerServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    servicer.set_server(server)
    print(f"[worker] started on port {port}  (hostname={hostname})")

    # Detect device and available memory for registration metadata
    device     = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    memory_mb  = _get_available_memory_mb(device)
    node_info  = NodeInfo(
        node_id=hostname, address=node_address, device=device, memory_mb=memory_mb
    )

    # Register with the coordinator (retries until coordinator is up)
    print(f"[discovery] registering with coordinator at {coordinator_addr} ...")
    try:
        result = announce_to_coordinator(coordinator_addr, node_info)
        print(f"[discovery] registered: run_id={result.run_id}, "
              f"worker_index={result.worker_index}")
    except RuntimeError as e:
        print(f"[discovery] FATAL: {e}")
        server.stop(0)
        sys.exit(1)

    server.wait_for_termination()
    print(f"[worker] {hostname} terminated cleanly")


if __name__ == '__main__':
    serve()
