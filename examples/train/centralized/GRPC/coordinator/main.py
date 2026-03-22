import io
import sys
import os
import threading
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import grpc
import torch
from torch import nn
from concurrent import futures
from torch.utils.data import DataLoader, TensorDataset

from proto_common.coordinator import coordinator_service_pb2_grpc
from proto_common.coordinator import coordinator_service_pb2
from proto_common.worker import worker_service_pb2_grpc
from proto_common.worker import worker_service_pb2

from torchslicer.core.slicer import Slicer
from torchslicer.strategies.uniform import UniformSplitter
from torchslicer.core.model_graph import ModelGraph


# ── helpers ────────────────────────────────────────────────────────────────────

_TORCH_TO_DTYPE = {
    torch.float32:  worker_service_pb2.FLOAT32,
    torch.float16:  worker_service_pb2.FLOAT16,
    torch.bfloat16: worker_service_pb2.BFLOAT16,
    torch.float64:  worker_service_pb2.FLOAT64,
    torch.int64:    worker_service_pb2.INT64,
    torch.int32:    worker_service_pb2.INT32,
}


def serialize_tensor(t: torch.Tensor) -> worker_service_pb2.Tensor:
    buf = io.BytesIO()
    torch.save(t, buf)
    return worker_service_pb2.Tensor(
        data=buf.getvalue(),
        shape=list(t.shape),
        dtype=_TORCH_TO_DTYPE.get(t.dtype, worker_service_pb2.FLOAT32),
    )


def build_layer_configs(layers: list) -> list:
    """Serialise a list of nn.Module layers into LayerConfig protos."""
    configs = []
    for layer in layers:
        buf = io.BytesIO()
        torch.save(layer, buf)
        configs.append(worker_service_pb2.LayerConfig(
            layer_type=layer.__class__.__name__,
            serialized=buf.getvalue(),
        ))
    return configs


def build_optimizer_config(cfg: dict) -> worker_service_pb2.OptimizerConfig:
    extra = {k: v for k, v in cfg.get("params", {}).items() if k != "lr"}
    buf = io.BytesIO()
    torch.save(extra, buf)
    return worker_service_pb2.OptimizerConfig(
        name=cfg["name"],
        lr=float(cfg["params"].get("lr", 0.001)),
        extra_params=buf.getvalue(),
    )


def build_criterion_config(cfg: dict) -> worker_service_pb2.CriterionConfig:
    buf = io.BytesIO()
    torch.save(cfg.get("params", {}), buf)
    return worker_service_pb2.CriterionConfig(
        name=cfg["name"],
        extra_params=buf.getvalue(),
    )


# ── Device ─────────────────────────────────────────────────────────────────────

class Device:
    def __init__(self, name: str, address: str, port: str):
        self.name = name
        self.address = address
        self.port = port
        self._stub = None

    def url(self) -> str:
        return f"{self.address}:{self.port}"

    def connect(self):
        self._stub = worker_service_pb2_grpc.WorkerServiceStub(
            grpc.insecure_channel(self.url()))

    def stub(self) -> worker_service_pb2_grpc.WorkerServiceStub:
        return self._stub


# ── Dispatcher ─────────────────────────────────────────────────────────────────

class Dispatcher(coordinator_service_pb2_grpc.CoordinatorServiceServicer):

    def __init__(self, network, criterion_cfg, optimizer_cfg, coordinator_addr,
                 max_epochs=1, gen_dataset=None, cluster: list[Device] = []):
        super().__init__()

        self.train_loader, _, _ = gen_dataset() if gen_dataset else (None, None, None)
        self.max_epochs = max_epochs
        self.current_epoch = 0
        self.current_batch = 0
        self.cluster = cluster
        self.coordinator_addr = coordinator_addr

        # Slice model
        slicer = Slicer(network)
        graph = ModelGraph.from_sequential(network)
        parts = UniformSplitter().split(graph, n_partitions=len(cluster))
        cut = parts[0].layer_indices[-1] + 1

        opt_cfg = build_optimizer_config(optimizer_cfg)
        crit_cfg = build_criterion_config(criterion_cfg)
        n = len(cluster)

        for i, dev in enumerate(cluster):
            dev.connect()
            is_last = (i == n - 1)
            layer_slice = slicer.layers[:cut] if i == 0 else slicer.layers[cut:]
            layer_cfgs = build_layer_configs(layer_slice)
            cfg = worker_service_pb2.SliceConfig(
                layers=layer_cfgs,
                optimizer=opt_cfg,
                criterion=crit_cfg if is_last else None,
                is_last=is_last,
                prev_worker=cluster[i - 1].url() if i > 0 else "",
                next_worker=cluster[i + 1].url() if i < n - 1 else "",
                coordinator=coordinator_addr,
            )
            for attempt in range(10):
                try:
                    res = dev.stub().init(cfg)
                    print(f"[init] {dev.name}: ok={res.ok}  {res.message}  ({res.hostname})")
                    break
                except grpc.RpcError:
                    print(f"Worker {dev.name} not ready, retry {attempt + 1}/10 ...")
                    time.sleep(3)

        self.init_time = time.time()
        threading.Timer(1.0, self._kick).start()

    def _kick(self):
        self.batch_done(coordinator_service_pb2.BatchDoneRequest(batch_id=0), None)

    # ── RPCs ───────────────────────────────────────────────────────────────────

    def report_metrics(self, request, context):
        print(f"[epoch {self.current_epoch} | batch {self.current_batch}] "
              f"loss={request.loss:.4f}")
        return coordinator_service_pb2.Empty()

    def batch_done(self, request, context):
        try:
            if self.current_epoch == self.max_epochs:
                print(f"Train FINISHED!  {time.time() - self.init_time:.1f}s")
                return coordinator_service_pb2.Empty()

            if self.current_batch < len(self.train_loader) - 1:
                self.current_batch += 1
            else:
                self.current_batch = 0
                self.current_epoch += 1

            inputs, labels = next(iter(self.train_loader))
            batch_id = self.current_epoch * len(self.train_loader) + self.current_batch
            threading.Thread(
                target=self._send_batch, args=(batch_id, inputs, labels)).start()
        except Exception as e:
            print(f"[batch_done] ERROR: {e}")
        return coordinator_service_pb2.Empty()

    def _send_batch(self, batch_id: int, inputs: torch.Tensor, labels: torch.Tensor):
        # Send label to last worker
        self.cluster[-1].stub().forward(worker_service_pb2.ForwardRequest(
            batch_id=batch_id,
            label=serialize_tensor(labels),
        ))
        # Send input to first worker
        self.cluster[0].stub().forward(worker_service_pb2.ForwardRequest(
            batch_id=batch_id,
            input=serialize_tensor(inputs),
        ))


# ── entrypoint ─────────────────────────────────────────────────────────────────

def get_dataset(batch_size=32):
    X = torch.randn(512, 32)
    y = torch.randint(0, 2, (512,))
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    return loader, loader, loader


def serve():
    port = sys.argv[1] if len(sys.argv) > 1 else "50054"
    coordinator_addr = f"coordinator:{port}"

    criterion_cfg = {"name": "CrossEntropyLoss", "params": {}}
    optimizer_cfg = {"name": "Adam", "params": {"lr": 0.001, "weight_decay": 0.0001}}
    cluster = [
        Device("worker1", "worker1", "50051"),
        Device("worker2", "worker2", "50051"),
    ]

    model = nn.Sequential(
        nn.Linear(32, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 2),
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    coordinator_service_pb2_grpc.add_CoordinatorServiceServicer_to_server(
        Dispatcher(model, criterion_cfg, optimizer_cfg, coordinator_addr,
                   max_epochs=3, gen_dataset=get_dataset, cluster=cluster),
        server)
    server.add_insecure_port('[::]:' + port)
    print(f"Coordinator started at port {port}")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
