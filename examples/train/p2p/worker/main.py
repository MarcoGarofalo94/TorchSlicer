"""
P2P worker entry point: every node runs this script (no coordinator process).

  IS_DRIVER=true   driver = worker 0: holds the data and slice 0, plans the split,
                   ships the other slices and drives training (``ts.P2PExecutor``).
  IS_DRIVER=false  follower: a plain worker (``ts.run_worker(coordinator_addr=None)``)
                   that waits for its slice from the driver.

Raw inputs never leave the driver, and labels go only to the last worker. A dead
follower fails the run (``fault_tolerance.detect_failures``), or, with
``fault_tolerance.enabled``, the run continues on the survivors.

Environment variables:
  IS_DRIVER          true/false (default false)
  WORKER_PEERS       comma-separated "host:port" list in slice order, driver first
                     (overrides discovery.peers from YAML)
  WORKER_ADDRESS     address advertised to peers (default hostname:port)
  EXPERIMENT_CONFIG  path to YAML experiment config

Customize ``build_model`` / ``get_dataset`` / ``get_pack`` / ``get_split`` below.
"""

import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import torchslicer as ts


# ── dataset / model (customize per experiment) ─────────────────────────────────

def get_dataset(data_dir: str = '/workspace/data', batch_size: int = None,
                n_train: int = None):
    import torchvision
    import torchvision.transforms as T
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
    return DataLoader(Subset(ds, indices), batch_size=batch_size,
                      shuffle=True, num_workers=0)


def build_model():
    import torchvision
    model         = torchvision.models.resnet18()
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, 10)
    return model


def get_pack():
    """Optional ``pack=`` function for ``ts.slice`` (e.g. ``examples.pack.gpt2.pack_gpt2``
    for HF LLMs).  None = trace the model with torch.fx."""
    return None


def get_split() -> dict:
    """Extra ``ts.slice`` arguments: ``strategy``, ``cuts``, ``expand``, ``example_input``."""
    return {"strategy": os.environ.get("SPLIT_STRATEGY", "uniform")}


# ── entry point ────────────────────────────────────────────────────────────────

def serve() -> list:
    """Run this node (role from IS_DRIVER); the driver returns the history."""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else None
    return ts.run_p2p(build_model, get_dataset, port=port, pack=get_pack(), **get_split())


if __name__ == '__main__':
    serve()
