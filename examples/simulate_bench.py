#!/usr/bin/env python3
"""
ts.simulate() benchmark — spawns N worker subprocesses locally.
Compares local single-process vs simulate with gRPC and TCP, on GPU if available.
"""
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchslicer as ts

torch.manual_seed(42)
N_SAMPLES = 2048
BATCH = 64
EPOCHS = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

X = torch.randn(N_SAMPLES, 64, device=DEVICE)
y = torch.randint(0, 10, (N_SAMPLES,), device=DEVICE)
loader = DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=False)


def make_model():
    return nn.Sequential(
        nn.Linear(64, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64),  nn.ReLU(),
        nn.Linear(64, 10),
    ).to(DEVICE)


def run_local():
    print(f"── Local (single process, {DEVICE}) ──────────────────────────")
    t0 = time.perf_counter()
    history = ts.run(make_model(), loader, n=2, epochs=EPOCHS,
                     optimizer="adamw", criterion="cross_entropy")
    elapsed = time.perf_counter() - t0
    for i, h in enumerate(history, 1):
        print(f"  epoch {i}: loss={h['loss']:.4f}")
    print(f"  total: {elapsed:.2f}s  ({elapsed/EPOCHS:.2f}s/epoch)\n")
    return elapsed


def run_simulate(n_workers, transport="grpc"):
    devices = [DEVICE] * n_workers
    print(f"── simulate ({n_workers} workers, {transport.upper()}, {DEVICE}) ──────────────────────────")
    t0 = time.perf_counter()
    history = ts.simulate(make_model(), loader, n=n_workers, epochs=EPOCHS,
                          optimizer="adamw", criterion="cross_entropy",
                          devices=devices, transport=transport, verbose=False)
    elapsed = time.perf_counter() - t0
    for i, h in enumerate(history, 1):
        print(f"  epoch {i}: loss={h['loss']:.4f}")
    print(f"  total: {elapsed:.2f}s  ({elapsed/EPOCHS:.2f}s/epoch)\n")
    return elapsed


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    t_local = run_local()
    t_grpc2 = run_simulate(2, transport="grpc")
    t_tcp2  = run_simulate(2, transport="tcp")
    t_grpc4 = run_simulate(4, transport="grpc")
    t_tcp4  = run_simulate(4, transport="tcp")

    print("── Summary ─────────────────────────────────────────")
    print(f"  local ({DEVICE})          {t_local/EPOCHS:.2f}s/epoch")
    print(f"  simulate 2w gRPC      {t_grpc2/EPOCHS:.2f}s/epoch  ({t_grpc2/t_local:.1f}x vs local)")
    print(f"  simulate 2w TCP       {t_tcp2/EPOCHS:.2f}s/epoch  ({t_tcp2/t_local:.1f}x vs local)")
    print(f"  simulate 4w gRPC      {t_grpc4/EPOCHS:.2f}s/epoch  ({t_grpc4/t_local:.1f}x vs local)")
    print(f"  simulate 4w TCP       {t_tcp4/EPOCHS:.2f}s/epoch  ({t_tcp4/t_local:.1f}x vs local)")
