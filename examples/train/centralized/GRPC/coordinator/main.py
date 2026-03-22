import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import torchslicer as ts
from torchslicer.executors.distributed import DistributedExecutor
from torchslicer.monitor import tracer


def get_dataset(batch_size=32):
    X = torch.randn(512, 32)
    y = torch.randint(0, 2, (512,))
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    return loader


def serve():
    tracer.auto_configure_if_env()   # no-op unless OTEL_EXPORTER_OTLP_ENDPOINT is set
    port = sys.argv[1] if len(sys.argv) > 1 else "50054"

    model = nn.Sequential(
        nn.Linear(32, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 2),
    )

    workers = [
        {"name": "worker1", "address": "worker1", "port": "50051"},
        {"name": "worker2", "address": "worker2", "port": "50051"},
    ]
    optimizer_cfg = {"name": "Adam", "params": {"lr": 0.001, "weight_decay": 0.0001}}
    criterion_cfg = {"name": "CrossEntropyLoss", "params": {}}

    executor = DistributedExecutor(
        workers=workers,
        coordinator_addr=f"coordinator:{port}",
    )
    sliced = ts.slice(model, strategy="uniform", n=len(workers), executor=executor)
    sliced.train(get_dataset(), optimizer_cfg, criterion_cfg, epochs=3, verbose=True)


if __name__ == '__main__':
    serve()
