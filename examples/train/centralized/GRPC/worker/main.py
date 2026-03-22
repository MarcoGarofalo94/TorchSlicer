import io
import sys
import threading
import socket

import grpc
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


def deserialize_tensor(msg: worker_service_pb2.Tensor) -> torch.Tensor:
    return torch.load(io.BytesIO(msg.data), weights_only=False)


def serialize_tensor(t: torch.Tensor) -> worker_service_pb2.Tensor:
    buf = io.BytesIO()
    torch.save(t, buf)
    return worker_service_pb2.Tensor(
        data=buf.getvalue(),
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

        # Per-batch state, keyed by batch_id.
        # Both dicts are always accessed under self._lock.
        self._labels: dict[int, torch.Tensor] = {}
        self._outputs: dict[int, torch.Tensor] = {}
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
            print(f"       param_mb={param_mb}  cuda_alloc_mb={cuda_alloc_mb}")

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
                pass  # span records topology info; work already done above

            return worker_service_pb2.StatusMessage(
                ok=True, message="Initialized", hostname=socket.gethostname())
        except Exception as e:
            print(f"[init] ERROR: {e}")
            return worker_service_pb2.StatusMessage(
                ok=False, message=str(e), hostname=socket.gethostname())

    # ── forward / backward RPCs (fire-and-forget) ──────────────────────────────

    def forward(self, request, context):
        threading.Thread(target=self._forward, args=(request,), daemon=True).start()
        return worker_service_pb2.Ack(batch_id=request.batch_id)

    def backward(self, request, context):
        threading.Thread(target=self._backward, args=(request,), daemon=True).start()
        return worker_service_pb2.Ack(batch_id=request.batch_id)

    # ── internal forward ───────────────────────────────────────────────────────

    def _forward(self, request: worker_service_pb2.ForwardRequest):
        batch_id = request.batch_id

        if self.is_last:
            self._forward_last(batch_id, request)
            return

        # Non-last: run forward, send activation to next worker
        tensor = deserialize_tensor(request.input).to(self.device)
        with _tracer.span(
            "torchslicer.worker.forward",
            batch_id=batch_id,
            worker=socket.gethostname(),
            input_shape=str(tuple(tensor.shape)),
        ) as s:
            out = self.layer(tensor)
            if s:
                s.set_attribute("output_shape", str(tuple(out.shape)))
        with self._lock:
            self._outputs[batch_id] = out

        stub = worker_service_pb2_grpc.WorkerServiceStub(_channel(self.next_worker))
        stub.forward(worker_service_pb2.ForwardRequest(
            batch_id=batch_id,
            input=serialize_tensor(out),
        ))

    def _forward_last(self, batch_id: int, request: worker_service_pb2.ForwardRequest):
        """
        The last worker receives two ForwardRequests per batch:
          1. from coordinator: label only (input.data is empty)
          2. from previous worker: activation only (label.data is empty)
        Both paths store their payload and trigger backward once both arrive.
        """
        is_label_msg = bool(request.label.data)

        if is_label_msg:
            label = deserialize_tensor(request.label).to(self.device)
            with self._lock:
                self._labels[batch_id] = label
                out = self._outputs.pop(batch_id, None)
        else:
            tensor = deserialize_tensor(request.input).to(self.device)
            out = self.layer(tensor)
            with self._lock:
                label = self._labels.pop(batch_id, None)
                if label is None:
                    self._outputs[batch_id] = out

        if out is not None and label is not None:
            self._run_backward_last(batch_id, out, label)

    # ── internal backward ──────────────────────────────────────────────────────

    def _backward(self, request: worker_service_pb2.BackwardRequest):
        batch_id = request.batch_id
        grad_in = deserialize_tensor(request.gradient).to(self.device)

        with self._lock:
            out = self._outputs.pop(batch_id, None)

        if out is None:
            print(f"[backward] WARNING: no cached output for batch_id={batch_id}")
            return

        with _tracer.span(
            "torchslicer.worker.backward",
            batch_id=batch_id,
            worker=socket.gethostname(),
            is_last=False,
        ):
            grad = self.layer.backward(prev_g=grad_in, out=out)
            self.layer.optimize()
        self._send_backward(batch_id, grad)

    def _run_backward_last(self, batch_id: int, out: torch.Tensor, label: torch.Tensor):
        with _tracer.span(
            "torchslicer.worker.forward",
            batch_id=batch_id,
            worker=socket.gethostname(),
            is_last=True,
            input_shape=str(tuple(out.shape)),
        ):
            loss = self.loss_fn(out, label)

        coord = coordinator_service_pb2_grpc.CoordinatorServiceStub(_channel(self.coordinator))
        coord.report_metrics(coordinator_service_pb2.MetricsMessage(
            batch_id=batch_id, loss=loss.item(), worker=socket.gethostname()))

        with _tracer.span(
            "torchslicer.worker.backward",
            batch_id=batch_id,
            worker=socket.gethostname(),
            is_last=True,
            loss=loss.item(),
        ):
            grad = self.layer.backward(loss=loss)
            self.layer.optimize()
        self._send_backward(batch_id, grad)

    def _send_backward(self, batch_id: int, grad: torch.Tensor):
        if self.prev_worker:
            stub = worker_service_pb2_grpc.WorkerServiceStub(_channel(self.prev_worker))
            stub.backward(worker_service_pb2.BackwardRequest(
                batch_id=batch_id,
                gradient=serialize_tensor(grad),
            ))
        else:
            coord = coordinator_service_pb2_grpc.CoordinatorServiceStub(
                _channel(self.coordinator))
            coord.batch_done(coordinator_service_pb2.BatchDoneRequest(
                batch_id=batch_id))

    # ── helpers ────────────────────────────────────────────────────────────────

    def _build_layers(self, layer_configs) -> list:
        return [
            torch.load(io.BytesIO(lc.serialized), weights_only=False)
            for lc in layer_configs
        ]


# ── entrypoint ─────────────────────────────────────────────────────────────────

def serve():
    _tracer.auto_configure_if_env()   # no-op unless OTEL_EXPORTER_OTLP_ENDPOINT is set
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
