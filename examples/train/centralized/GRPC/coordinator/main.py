import argparse
import os
import signal
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

import torchslicer as ts
from torchslicer.executors.distributed import DistributedExecutor
from torchslicer.discovery import CoordinatorDiscovery, StaticDiscovery
from torchslicer.config import RunConfig
from torchslicer.monitor import tracer


def get_dataset(data_dir='/workspace/data', batch_size=None, n_train=None):
    batch_size = batch_size or int(os.environ.get("BATCH_SIZE", 64))
    n_train    = n_train    or int(os.environ.get("N_TRAIN",    10000))
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)
    indices = torch.randperm(len(ds))[:n_train].tolist()
    loader  = DataLoader(
        Subset(ds, indices), batch_size=batch_size, shuffle=True, num_workers=0)
    return loader


def build_model():
    # ResNet18 adapted for CIFAR-10 (32×32 input, no downsampling in first layer)
    model         = torchvision.models.resnet18()
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, 10)
    return model


def serve():
    tracer.auto_configure_if_env()

    parser = argparse.ArgumentParser(description="TorchSlicer coordinator")
    parser.add_argument("port", nargs="?", default="50054",
                        help="Port for the coordinator gRPC server (default: 50054)")
    parser.add_argument("--config", default=None,
                        help="Path to YAML experiment config (overrides env vars)")
    args = parser.parse_args()

    # Load config: YAML (if provided) > env vars > defaults
    cfg = RunConfig.load(args.config)

    print(f"[coordinator] run_id={cfg.run_id}  n_workers={cfg.discovery.n_workers}  "
          f"epochs={cfg.training.epochs}  gpipe={cfg.pipeline.use_gpipe}  "
          f"n_micro={cfg.pipeline.n_micro}  checkpoint={cfg.checkpoint.enabled}")

    coordinator_addr = f"coordinator:{args.port}"
    n = cfg.discovery.n_workers

    # Select discovery backend from config
    if cfg.discovery.backend == "static":
        if not cfg.discovery.peers:
            raise ValueError(
                "discovery.backend=static requires discovery.peers to be set "
                "(WORKER_PEERS env var or peers list in YAML)"
            )
        discovery = StaticDiscovery(peers=cfg.discovery.peers)
    else:
        # Default: coordinator-based registration
        discovery = CoordinatorDiscovery(run_id=cfg.run_id)

    executor = DistributedExecutor(
        discovery=discovery,
        coordinator_addr=coordinator_addr,
        run_config=cfg,
    )

    sliced = ts.slice(build_model(), strategy="uniform", n=n, executor=executor)
    sliced.train(
        get_dataset(),
        cfg.training.optimizer,
        cfg.training.criterion,
        epochs          = cfg.training.epochs,
        verbose         = True,
        mixed_precision = cfg.training.mixed_precision,
        use_gpipe       = cfg.pipeline.use_gpipe,
        n_micro_batches = cfg.pipeline.n_micro,
        run_config      = cfg,
    )

    run_dir = os.path.join(cfg.logging.dir, cfg.run_id) if cfg.logging.enabled else ""
    print(f"[coordinator] run complete — waiting for shutdown signal (SIGTERM/SIGINT)")
    if run_dir:
        print(f"[coordinator] logs → {run_dir}/run_manifest.json")
        print(f"[coordinator] metrics → {run_dir}/metrics.jsonl")
    signal.pause()


if __name__ == '__main__':
    serve()
