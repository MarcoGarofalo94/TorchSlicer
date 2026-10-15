"""One script for every environment: a small CNN on synthetic CIFAR-shaped data.

The same file runs in one process, as OS processes, in compose, on k8s or on bare
metal; the config (``deploy.topology``) and each node's ``ROLE`` decide what it does:

    python examples/run_anywhere/train.py                               # local, one process
    EXPERIMENT_CONFIG=experiments/run_anywhere_centralized.yaml \\
        python examples/run_anywhere/train.py                           # every node in-process
    python -m torchslicer launch examples/run_anywhere/train.py \\
        -c experiments/run_anywhere_centralized.yaml --target processes # one process per node
    python -m torchslicer launch ... --target compose|k8s|baremetal     # render a deployment

See docs/deploy/README.md. The data is synthetic (fixed seed), so every target trains
on the same batches from the same weights, and the losses must match.
"""

import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import torchslicer as ts

SEED = 0


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return self.fc(torch.flatten(x, 1))


def build_model():
    torch.manual_seed(SEED)
    return SmallCNN()


def get_loader():
    g = torch.Generator().manual_seed(SEED + 1)
    x = torch.randn(128, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (128,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=32)


if __name__ == "__main__":
    history = ts.run(build_model, get_loader)
    if history:   # only the node that drives training has one
        print("HISTORY " + json.dumps([h["loss"] for h in history]), flush=True)
