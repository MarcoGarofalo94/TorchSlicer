"""Node script for bench_nat.py: a 3-slice MLP whose cuts carry 1 MiB per batch and
direction (batch 256 x 1024 float32), U-shaped P2P, synthetic data."""
import json
import os
import time

import torch
import torch.nn as nn

import torchslicer as ts

B, D = 256, 1024


def build_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 10))


def get_loader():
    g = torch.Generator().manual_seed(1)
    n = int(os.environ.get("BENCH_BATCHES", 32))
    return [(torch.randn(B, D, generator=g), torch.randint(0, 10, (B,), generator=g))
            for _ in range(n)]


class EpochTimer(ts.TrainingCallback):
    def __init__(self):
        self.seconds, self._t0 = [], 0.0

    def on_epoch_begin(self, epoch):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, metrics):
        self.seconds.append(time.perf_counter() - self._t0)
        return metrics


if __name__ == "__main__":
    torch.set_num_threads(1)
    timer = EpochTimer()
    history = ts.run(build_model, get_loader, cuts=["1", "3"], callbacks=[timer])
    if history:
        for h, s in zip(history, timer.seconds):
            h["epoch_s"] = s
        with open(os.environ["BENCH_OUT"], "w") as f:
            json.dump(history, f)
