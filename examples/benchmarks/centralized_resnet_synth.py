#!/usr/bin/env python3
import argparse
import os
import signal

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, TensorDataset

import torchslicer as ts
from torchslicer.config import RunConfig
from torchslicer.discovery import CoordinatorDiscovery, StaticDiscovery
from torchslicer.executors.distributed import DistributedExecutor
from torchslicer.monitor import tracer


def _build_model() -> nn.Module:
    model = torchvision.models.resnet18()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model


def _get_loader() -> DataLoader:
    batch_size = int(os.environ.get("BATCH_SIZE", "64"))
    n_train = int(os.environ.get("N_TRAIN", "4096"))
    seed = int(os.environ.get("SEED", "1234"))
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_train, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (n_train,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False, num_workers=0)


def serve() -> None:
    tracer.auto_configure_if_env()

    parser = argparse.ArgumentParser(description="TorchSlicer synthetic ResNet benchmark coordinator")
    parser.add_argument("port", nargs="?", default="50054")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = RunConfig.load(args.config)
    print(
        f"[coordinator] run_id={cfg.run_id} n_workers={cfg.discovery.n_workers} "
        f"epochs={cfg.training.epochs} gpipe={cfg.pipeline.use_gpipe} "
        f"n_micro={cfg.pipeline.n_micro} synthetic_resnet=true"
    )

    coordinator_addr = os.environ.get("COORDINATOR_ADDRESS", f"coordinator:{args.port}")
    coordinator_bind_addr = os.environ.get("COORDINATOR_BIND_ADDRESS", f"0.0.0.0:{args.port}")

    if cfg.discovery.backend == "static":
        if not cfg.discovery.peers:
            raise ValueError("discovery.backend=static requires discovery.peers")
        discovery = StaticDiscovery(peers=cfg.discovery.peers)
    else:
        discovery = CoordinatorDiscovery(run_id=cfg.run_id)

    executor = DistributedExecutor(
        discovery=discovery,
        coordinator_addr=coordinator_addr,
        coordinator_bind_addr=coordinator_bind_addr,
        run_config=cfg,
    )

    sliced = ts.slice(_build_model(), strategy="uniform", n=cfg.discovery.n_workers, executor=executor)
    sliced.train(
        _get_loader(),
        cfg.training.optimizer,
        cfg.training.criterion,
        epochs=cfg.training.epochs,
        verbose=True,
        mixed_precision=cfg.training.mixed_precision,
        use_gpipe=cfg.pipeline.use_gpipe,
        n_micro_batches=cfg.pipeline.n_micro,
        run_config=cfg,
    )

    run_dir = os.path.join(cfg.logging.dir, cfg.run_id) if cfg.logging.enabled else ""
    keep_alive = os.environ.get("KEEP_ALIVE", "1").strip().lower() not in {"0", "false", "no"}
    if not keep_alive:
        print("[coordinator] run complete — exiting immediately (KEEP_ALIVE disabled)")
        if run_dir:
            print(f"[coordinator] metrics -> {run_dir}/metrics.jsonl")
        return

    print("[coordinator] run complete — waiting for shutdown signal")
    signal.pause()


if __name__ == "__main__":
    serve()
