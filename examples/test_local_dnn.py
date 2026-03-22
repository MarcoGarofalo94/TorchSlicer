#!/usr/bin/env python3
"""
Quick smoke-test for TorchSlicer LocalExecutor with a simple DNN.
Synthetic data, no downloads required.
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchslicer as ts

# ── model ──────────────────────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
)

# ── synthetic dataset (600 samples, 64-dim input, 10 classes) ─────────────────
torch.manual_seed(42)
X = torch.randn(600, 64)
y = torch.randint(0, 10, (600,))
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

# ── slice into 4 partitions (LocalExecutor by default) ────────────────────────
N_PARTITIONS = 4
sliced = ts.slice(model, strategy="uniform", n=N_PARTITIONS)

print(f"Model sliced into {N_PARTITIONS} partitions:")
graph_layers = sliced.graph.get_layers()
for p in sliced.partitions:
    names = [type(graph_layers[i]).__name__ for i in p.layer_indices]
    print(f"  Partition {p.index}: {names}")

# ── train ─────────────────────────────────────────────────────────────────────
optimizer_cfg  = {"name": "SGD",            "params": {"lr": 0.01, "momentum": 0.9}}
criterion_cfg  = {"name": "CrossEntropyLoss", "params": {}}

print("\nStarting training...\n")
history = sliced.train(loader, optimizer_cfg, criterion_cfg, epochs=5, verbose=True)

print("\n── Summary ──")
for i, metrics in enumerate(history, 1):
    print(f"  Epoch {i}: avg_loss={metrics['loss']:.4f}")
