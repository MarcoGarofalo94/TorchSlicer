import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

import torchslicer as ts
from torchslicer.executors.distributed import DistributedExecutor
from torchslicer.monitor import tracer


def get_dataset(data_dir='/workspace/data', batch_size=64, n_train=10000):
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    indices = torch.randperm(len(ds))[:n_train].tolist()
    loader = DataLoader(Subset(ds, indices), batch_size=batch_size, shuffle=True, num_workers=0)
    return loader


def build_model():
    # ResNet18 adapted for CIFAR-10 (32x32 input, no downsampling in first layer)
    model = torchvision.models.resnet18()
    model.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc     = nn.Linear(512, 10)
    return model


def serve():
    tracer.auto_configure_if_env()
    port = sys.argv[1] if len(sys.argv) > 1 else "50054"

    workers = [
        {"name": "worker1", "address": "worker1", "port": "50051"},
        {"name": "worker2", "address": "worker2", "port": "50051"},
    ]
    optimizer_cfg = {"name": "SGD",  "params": {"lr": 0.05, "momentum": 0.9, "weight_decay": 5e-4}}
    criterion_cfg = {"name": "CrossEntropyLoss", "params": {}}

    executor = DistributedExecutor(
        workers=workers,
        coordinator_addr=f"coordinator:{port}",
    )
    sliced = ts.slice(build_model(), strategy="uniform", n=len(workers), executor=executor)
    sliced.train(get_dataset(), optimizer_cfg, criterion_cfg, epochs=20, verbose=True)


if __name__ == '__main__':
    serve()
