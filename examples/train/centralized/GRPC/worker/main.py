import io
import sys
import threading
import socket

import grpc
import numpy as np
import torch
from torch import nn
from torch import optim
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
from torchslicer.monitor import tracer as _tracer


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


# ── WorkerServicer ─────────────────────────────────────────────────────────────

class WorkerServicer(worker_service_pb2_grpc.WorkerServiceServicer):

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layer: SplitLayer = None
        self.loss_fn = None
        self.is_last = False
        self.prev_worker: str = None
        self.next_worker: str = None
        self.coordinator: str = None
        self._n_micro: int = 1

        # Persistent stubs — created once in init(), reused across all batches
        self._next_stub = None
        self._prev_stub = None
        self._coord_stub = None

        # Single-worker compute pool: serialises all forward/backward ops on
        # this worker so concurrent micro-batch RPCs never race on self.layer.x
        # or the model's .grad tensors.  Pipeline parallelism still happens
        # because DIFFERENT workers run their compute pools concurrently.
        self._pool = futures.ThreadPoolExecutor(max_workers=1)

        # Per-batch state, keyed by batch_id.
        # _outputs stores (out_tensor, x_ref) tuples so each backward uses the
        # correct cut-point tensor even after subsequent forwards overwrite self.layer.x.
        self._labels:  dict[int, torch.Tensor]                    = {}
        self._outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # GPipe: accumulated unscaled losses keyed by full_batch_id (last worker)
        self._micro_losses: dict[int, float] = {}
        self._lock = threading.Lock()

    # ── init ───────────────────────────────────────────────────────────────────

    def init(self, request, context):
        try:
            layers = self._build_layers(request.layers)
            self.layer = SplitLayer(layers, is_last=request.is_last)
            self.is_last = request.is_last

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
            self._n_micro = max(1, request.n_micro) if request.n_micro else 1

            # Build persistent stubs now so forward/backward never open a channel
            if self.next_worker:
                self._next_stub = worker_service_pb2_grpc.WorkerServiceStub(_channel(self.next_worker))
            if self.prev_worker:
                self._prev_stub = worker_service_pb2_grpc.WorkerServiceStub(_channel(self.prev_worker))
            self._coord_stub = coordinator_service_pb2_grpc.CoordinatorServiceStub(_channel(self.coordinator))

            self.layer = self.layer.to(self.device)
            if self.loss_fn:
                self.loss_fn = self.loss_fn.to(self.device)
            layer_names = [type(l).__name__ for l in self.layer.layers]

            param_bytes = sum(p.numel() * p.element_size() for p in self.layer.parameters())
            param_mb    = round(param_bytes / (1024 * 1024), 2)
            cuda_alloc_mb = round(
                torch.cuda.memory_allocated(self.device) / (1024 * 1024), 2
            ) if self.device.type == 'cuda' else 0.0

            print(f"[init] layers={layer_names}  is_last={self.is_last}  device={self.device}")
            print(f"       prev={self.prev_worker}  next={self.next_worker}")
            print(f"       param_mb={param_mb}  cuda_alloc_mb={cuda_alloc_mb}  n_micro={self._n_micro}")

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
            return worker_service_pb2.StatusMessage(
                ok=False, message=str(e), hostname=socket.gethostname())

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

            if self.is_last:
                self._forward_last(batch_id, request)
                return

            # Non-last: run forward, save (out, x_ref), send activation to next worker
            tensor = deserialize_tensor(request.input).to(self.device)
            with _tracer.span(
                "torchslicer.worker.forward",
                batch_id=batch_id,
                worker=socket.gethostname(),
                input_shape=str(tuple(tensor.shape)),
            ) as s:
                out = self.layer(tensor)
                x_ref = self.layer.x   # save cut-point before next forward can overwrite it
                if s:
                    s.set_attribute("output_shape", str(tuple(out.shape)))

            with self._lock:
                self._outputs[batch_id] = (out, x_ref)

            self._next_stub.forward(worker_service_pb2.ForwardRequest(
                batch_id=batch_id,
                input=serialize_tensor(out),
            ))
        except Exception as e:
            print(f"[forward] ERROR batch_id={request.batch_id}: {e}")
            import traceback; traceback.print_exc()

    def _forward_last(self, batch_id: int, request: worker_service_pb2.ForwardRequest):
        """
        Last worker receives two messages per micro-batch:
          1. coordinator → label only
          2. previous worker → activation only
        Whichever arrives second triggers forward+backward.
        """
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
                    out = self.layer(tensor)
                    x_ref = self.layer.x   # save cut-point immediately

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
            batch_id  = request.batch_id
            n_micro   = self._n_micro
            is_last_micro = (n_micro <= 1) or (batch_id % n_micro == n_micro - 1)

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
                # Propagate gradient through this partition; use saved x_ref
                # so we get the correct cut-point gradient even after subsequent
                # micro-batch forwards overwrote self.layer.x
                out.backward(grad_in)
                grad = x_ref.grad
                if is_last_micro:
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
            n_micro   = self._n_micro
            is_last_micro = (n_micro <= 1) or (batch_id % n_micro == n_micro - 1)
            full_batch_id = batch_id // n_micro if n_micro > 1 else batch_id

            loss_unscaled = self.loss_fn(out, label)
            # Scale loss by 1/M so accumulated gradients match full-batch magnitude
            loss = loss_unscaled / n_micro

            with self._lock:
                self._micro_losses[full_batch_id] = (
                    self._micro_losses.get(full_batch_id, 0.0) + loss_unscaled.item()
                )

            if is_last_micro:
                avg_loss = self._micro_losses.pop(full_batch_id) / n_micro
                self._coord_stub.report_metrics(coordinator_service_pb2.MetricsMessage(
                    batch_id=batch_id, loss=avg_loss, worker=socket.gethostname()))

            with _tracer.span(
                "torchslicer.worker.backward",
                batch_id=batch_id,
                worker=socket.gethostname(),
                is_last=True,
                loss=loss_unscaled.item(),
            ):
                loss.backward()
                grad = x_ref.grad   # use saved x_ref, not self.layer.x
                if is_last_micro:
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
            self._prev_stub.backward(worker_service_pb2.BackwardRequest(
                batch_id=batch_id,
                gradient=serialize_tensor(grad),
            ))
        elif is_last_micro:
            # First worker signals batch_done only on the last micro-batch
            self._coord_stub.batch_done(coordinator_service_pb2.BatchDoneRequest(
                batch_id=batch_id))

    # ── helpers ────────────────────────────────────────────────────────────────

    def _build_layers(self, layer_configs) -> list:
        return [
            torch.load(io.BytesIO(lc.serialized), weights_only=False)
            for lc in layer_configs
        ]


# ── entrypoint ─────────────────────────────────────────────────────────────────

def serve():
    _tracer.auto_configure_if_env()
    port = sys.argv[1] if len(sys.argv) > 1 else "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=_GRPC_OPTS)
    worker_service_pb2_grpc.add_WorkerServiceServicer_to_server(
        WorkerServicer(), server)
    server.add_insecure_port('[::]:' + port)
    print(f"Worker started at port {port}")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
