import io
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field

import torch
from torch import nn

from .base import BaseExecutor
from ..monitor import tracer

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

def _channel(addr): return grpc.insecure_channel(addr, options=_GRPC_OPTS)


# ── tensor helpers ─────────────────────────────────────────────────────────────

_TORCH_TO_DTYPE = {}
_DTYPE_TO_TORCH = {}

def _init_dtype_maps():
    if not _GRPC_AVAILABLE:
        return
    mapping = [
        (torch.float32,  worker_service_pb2.FLOAT32),
        (torch.float16,  worker_service_pb2.FLOAT16),
        (torch.bfloat16, worker_service_pb2.BFLOAT16),
        (torch.float64,  worker_service_pb2.FLOAT64),
        (torch.int64,    worker_service_pb2.INT64),
        (torch.int32,    worker_service_pb2.INT32),
    ]
    for torch_t, proto_t in mapping:
        _TORCH_TO_DTYPE[torch_t] = proto_t
        _DTYPE_TO_TORCH[proto_t] = torch_t

_init_dtype_maps()


def _serialize_tensor(t: torch.Tensor):
    buf = io.BytesIO()
    torch.save(t, buf)
    return worker_service_pb2.Tensor(
        data=buf.getvalue(),
        shape=list(t.shape),
        dtype=_TORCH_TO_DTYPE.get(t.dtype, worker_service_pb2.FLOAT32),
    )


# ── config builders ────────────────────────────────────────────────────────────

def _build_layer_configs(layers: list) -> list:
    configs = []
    for layer in layers:
        buf = io.BytesIO()
        torch.save(layer, buf)
        configs.append(worker_service_pb2.LayerConfig(
            layer_type=layer.__class__.__name__,
            serialized=buf.getvalue(),
        ))
    return configs


def _build_optimizer_config(cfg: dict):
    extra = {k: v for k, v in cfg.get("params", {}).items() if k != "lr"}
    buf = io.BytesIO()
    torch.save(extra, buf)
    return worker_service_pb2.OptimizerConfig(
        name=cfg["name"],
        lr=float(cfg["params"].get("lr", 0.001)),
        extra_params=buf.getvalue(),
    )


def _build_criterion_config(cfg: dict):
    buf = io.BytesIO()
    torch.save(cfg.get("params", {}), buf)
    return worker_service_pb2.CriterionConfig(
        name=cfg["name"],
        extra_params=buf.getvalue(),
    )


# ── coordinator servicer (embedded gRPC server inside the executor) ────────────

class _CoordinatorServicer(
    coordinator_service_pb2_grpc.CoordinatorServiceServicer
    if _GRPC_AVAILABLE else object
):
    def __init__(self, executor: "DistributedExecutor"):
        self._executor = executor

    def batch_done(self, request, context):
        self._executor._batch_done.set()
        return coordinator_service_pb2.Empty()

    def report_metrics(self, request, context):
        with self._executor._lock:
            self._executor._batch_losses.append(request.loss)
        return coordinator_service_pb2.Empty()


# ── worker proxy ───────────────────────────────────────────────────────────────

@dataclass
class _WorkerProxy:
    name: str
    address: str
    port: str
    _stub: object = field(default=None, repr=False)

    def url(self) -> str:
        return f"{self.address}:{self.port}"

    def connect(self):
        self._stub = worker_service_pb2_grpc.WorkerServiceStub(_channel(self.url()))

    def stub(self):
        return self._stub


# ── DistributedExecutor ────────────────────────────────────────────────────────

class DistributedExecutor(BaseExecutor):
    """
    Centralized topology executor. This process acts as the coordinator:
      - Distributes model slices to remote workers via gRPC at setup() time.
      - Drives the training loop batch-by-batch; each batch is synchronous
        (train_epoch blocks until the last worker calls batch_done).

    Usage::

        executor = DistributedExecutor(
            workers=[
                {"name": "worker1", "address": "worker1", "port": "50051"},
                {"name": "worker2", "address": "worker2", "port": "50051"},
            ],
            coordinator_addr="coordinator:50054",
        )
        sliced = ts.slice(model, strategy="uniform", n=2, executor=executor)
        sliced.train(loader, optimizer_cfg, criterion_cfg, epochs=3, verbose=True)
    """

    def __init__(self, workers: list[dict], coordinator_addr: str):
        """
        workers: list of dicts with keys "name", "address", "port"
        coordinator_addr: "host:port" workers will use to call back (batch_done, report_metrics)
        """
        if not _GRPC_AVAILABLE:
            raise ImportError(
                "grpcio is required for DistributedExecutor. "
                "Install it with: pip install grpcio"
            )
        self._worker_cfgs = workers
        self.coordinator_addr = coordinator_addr

        self._proxies: list[_WorkerProxy] = []
        self._grpc_server = None

        # Synchronisation between gRPC callbacks and train_epoch loop
        self._batch_done = threading.Event()
        self._batch_losses: list[float] = []
        self._lock = threading.Lock()

    # ── BaseExecutor interface ─────────────────────────────────────────────────

    def setup(self, model_graph, partitions, optimizer_cfg: dict, criterion_cfg: dict) -> None:
        layers = model_graph.get_layers()
        n = len(partitions)

        # Start embedded coordinator gRPC server
        self._grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=_GRPC_OPTS)
        coordinator_service_pb2_grpc.add_CoordinatorServiceServicer_to_server(
            _CoordinatorServicer(self), self._grpc_server)
        port = self.coordinator_addr.split(":")[-1]
        self._grpc_server.add_insecure_port(f"[::]:{port}")
        self._grpc_server.start()

        # Build worker proxies and send slice configs
        opt_cfg = _build_optimizer_config(optimizer_cfg)
        crit_cfg = _build_criterion_config(criterion_cfg)

        # Build all proxies first so next_worker urls are available
        self._proxies = [
            _WorkerProxy(name=w["name"], address=w["address"], port=w["port"])
            for w in self._worker_cfgs
        ]
        for proxy in self._proxies:
            proxy.connect()

        for i, proxy in enumerate(self._proxies):
            is_last = (i == n - 1)
            partition_layers = [layers[j] for j in partitions[i].layer_indices]
            cfg = worker_service_pb2.SliceConfig(
                layers=_build_layer_configs(partition_layers),
                optimizer=opt_cfg,
                criterion=crit_cfg if is_last else None,
                is_last=is_last,
                prev_worker=self._proxies[i - 1].url() if i > 0 else "",
                next_worker=self._proxies[i + 1].url() if i < n - 1 else "",
                coordinator=self.coordinator_addr,
            )
            for attempt in range(10):
                try:
                    res = proxy.stub().init(cfg)
                    print(f"[init] {proxy.name}: ok={res.ok}  {res.message}  ({res.hostname})")
                    break
                except grpc.RpcError:
                    print(f"[init] {proxy.name} not ready, retry {attempt + 1}/10 ...")
                    time.sleep(3)

    def train_epoch(self, data_loader, epoch: int = 0, verbose: bool = False) -> dict:
        total_loss = 0.0
        n_batches = 0
        n_total = len(data_loader)

        with tracer.span("torchslicer.epoch", epoch=epoch, n_workers=len(self._proxies)):
            for inputs, labels in data_loader:
                batch_id = epoch * n_total + n_batches
                with self._lock:
                    self._batch_losses.clear()
                self._batch_done.clear()

                with tracer.span(
                    "torchslicer.batch",
                    epoch=epoch,
                    batch_id=batch_id,
                    batch_index=n_batches,
                    input_shape=str(tuple(inputs.shape)),
                ) as batch_span:
                    self._send_batch(batch_id, inputs, labels)
                    self._batch_done.wait()

                    with self._lock:
                        loss = sum(self._batch_losses) / len(self._batch_losses) \
                            if self._batch_losses else 0.0
                    if batch_span:
                        batch_span.set_attribute("loss", loss)

                total_loss += loss
                n_batches += 1

                if verbose:
                    print(f"  [epoch {epoch} | batch {n_batches}/{n_total}] loss={loss:.4f}")

        avg = total_loss / n_batches if n_batches > 0 else 0.0
        if verbose:
            print(f"[epoch {epoch}] avg_loss={avg:.4f}")
        return {"loss": avg}

    def teardown(self) -> None:
        if self._grpc_server:
            self._grpc_server.stop(grace=0)
            self._grpc_server = None
        self._proxies.clear()

    # ── internal ───────────────────────────────────────────────────────────────

    def _send_batch(self, batch_id: int, inputs: torch.Tensor, labels: torch.Tensor):
        # Label goes directly to last worker
        self._proxies[-1].stub().forward(worker_service_pb2.ForwardRequest(
            batch_id=batch_id,
            label=_serialize_tensor(labels),
        ))
        # Input goes to first worker, which chains forward through the slice pipeline
        self._proxies[0].stub().forward(worker_service_pb2.ForwardRequest(
            batch_id=batch_id,
            input=_serialize_tensor(inputs),
        ))
