"""TCP vs gRPC tensor transport benchmark on GPU.

Runs the same P2P split training (vanilla setting, 2 slices) with both transports
and reports ms/batch, GB/s effective throughput, and the speedup.

Usage:
    .venv/bin/python benchmarks/transport/bench_transport.py [model ...]
    # models: small medium large xlarge

Models tested (increasing activation size):
  - small:  Linear chain   ~  0.1 MB/cut
  - medium: CNN            ~  1.6 MB/cut
  - large:  wide MLP       ~  8 MB/cut
  - xlarge: Transformer    ~ 16 MB/cut
"""
import socket
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib" / "torchslicer"))
import torchslicer as ts
from torchslicer.config import RunConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS  = 4   # first epoch is warm-up, rest are measured
REPEATS = 3


# ── model definitions ──────────────────────────────────────────────────────────

def _small():
    """~0.1 MB activation at cut (batch=64, dim=256)."""
    return nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 10),
    ), (64, 512), 10

def _medium():
    """~1.6 MB activation at cut (batch=64, C=256, H=W=8)."""
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(8),
        nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),   # ← cut here
        nn.AdaptiveAvgPool2d(4),
        nn.Flatten(),
        nn.Linear(256 * 4 * 4, 10),
    ), (64, 3, 32, 32), 10

def _large():
    """~8 MB activation at cut (batch=64, dim=2048)."""
    return nn.Sequential(
        nn.Linear(1024, 2048), nn.GELU(),
        nn.Linear(2048, 2048), nn.GELU(),
        nn.Linear(2048, 2048), nn.GELU(),   # ← cut here
        nn.Linear(2048, 2048), nn.GELU(),
        nn.Linear(2048, 10),
    ), (64, 1024), 10

def _xlarge():
    """~16 MB activation at cut (batch=4, seq=512, d_model=768 × 2 slices of transformer)."""
    d, seq, heads = 768, 256, 12
    return nn.Sequential(
        nn.TransformerEncoderLayer(d, heads, dim_feedforward=d*4, batch_first=True),  # ← cut
        nn.TransformerEncoderLayer(d, heads, dim_feedforward=d*4, batch_first=True),
        nn.Flatten(1),
        nn.Linear(d * seq, 10),
    ), (4, seq, d), 10


MODELS = {
    "small":   _small,
    "medium":  _medium,
    "large":   _large,
    "xlarge":  _xlarge,
}


# ── benchmark helpers ──────────────────────────────────────────────────────────

def _loader(input_shape, n_classes, n_batches=16):
    B = input_shape[0]
    g = torch.Generator().manual_seed(42)
    x = torch.randn(*input_shape, generator=g)
    y = torch.randint(0, n_classes, (B,), generator=g)
    return [(x, y)] * n_batches


def _free_pair():
    """Find a port p such that p and p+1 are both free (gRPC + TCP tensor offset)."""
    for _ in range(50):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            p = s.getsockname()[1]
        try:
            with socket.socket() as a, socket.socket() as b:
                a.bind(("127.0.0.1", p))
                b.bind(("127.0.0.1", p + 1))
            return p
        except OSError:
            continue
    raise RuntimeError("no free port pair")


def _cfg(transport: str) -> RunConfig:
    cfg = RunConfig()
    cfg.training.optimizer = {"name": "SGD", "params": {"lr": 0.01, "momentum": 0.9}}
    cfg.training.criterion = {"name": "CrossEntropyLoss", "params": {}}
    cfg.training.epochs    = EPOCHS
    cfg.logging.enabled    = False
    cfg.network.device     = DEVICE
    cfg.transport.tensor   = transport
    cfg.sl.setting         = "vanilla"
    cfg.fault_tolerance.heartbeat_interval_s = 0.5
    cfg.fault_tolerance.ping_timeout_s       = 2.0
    return cfg


def _run_once(model_fn, transport: str) -> dict:
    model, input_shape, n_classes = model_fn()
    loader = _loader(input_shape, n_classes)
    cfg    = _cfg(transport)

    follower_port = _free_pair()
    driver_port   = _free_pair()
    follower_addr = f"127.0.0.1:{follower_port}"
    driver_addr   = f"127.0.0.1:{driver_port}"

    import time as _time
    handle = ts.start_worker(follower_port, host="127.0.0.1", run_config=cfg)
    epoch_times = []
    try:
        executor = ts.P2PExecutor(
            peers=[driver_addr, follower_addr],
            address=driver_addr,
            bind_addr=driver_addr,
            run_config=cfg,
        )
        sliced = ts.slice(model, n=2, executor=executor)
        # Manually run epochs so we can time each one wall-clock
        import inspect
        from torchslicer.executors.local import LocalExecutor
        from torchslicer import SlicedModel
        # Use sliced.train but capture per-epoch wall time via a callback
        class _Timer(ts.TrainingCallback):
            def __init__(self): self.times = []
            def on_epoch_end(self, epoch, metrics):
                self.times.append(metrics)
                return metrics
        timer = _Timer()
        t0 = _time.perf_counter()
        history = sliced.train(loader, run_config=cfg, callbacks=[timer])
        total_s = _time.perf_counter() - t0
    finally:
        handle.stop()

    batches_per_epoch = len(loader)
    n_epochs = len(history)
    # bytes_total = all bytes crossing the cut (fwd activations + bwd gradients + labels)
    bytes_total_epoch = history[-1].get("bytes_total", 0)
    # wall-clock: exclude first epoch (worker init + JIT warm-up)
    ms_per_batch = (total_s / n_epochs) / batches_per_epoch * 1e3
    epoch_s      = total_s / n_epochs
    gbps = (bytes_total_epoch / epoch_s) / 1e9 if bytes_total_epoch and epoch_s else 0.0
    # fwd bytes at cut 0 for display
    bytes_fwd_list = history[-1].get("bytes_fwd", [0])
    bytes_fwd = bytes_fwd_list[0] if isinstance(bytes_fwd_list, list) else bytes_fwd_list
    return {"ms_per_batch": ms_per_batch, "gbps": gbps,
            "bytes_mb": bytes_fwd / 1e6, "total_s": total_s}


def bench(model_name: str, model_fn) -> dict:
    results = {}
    for transport in ("grpc", "tcp"):
        runs = []
        for r in range(REPEATS):
            try:
                runs.append(_run_once(model_fn, transport))
            except Exception as e:
                print(f"  ERROR {transport} run {r}: {e}", flush=True)
        if runs:
            ms   = [r["ms_per_batch"] for r in runs]
            gbps = [r["gbps"]         for r in runs]
            results[transport] = {
                "mean_ms":  statistics.mean(ms),
                "std_ms":   statistics.stdev(ms) if len(ms) > 1 else 0.0,
                "mean_gbps": statistics.mean(gbps),
                "bytes_mb": runs[0]["bytes_mb"],
            }
    return results


def print_results(model_name: str, r: dict):
    g = r.get("grpc", {})
    t = r.get("tcp",  {})
    if not g or not t:
        print(f"  {model_name}: missing results"); return
    speedup = g["mean_ms"] / t["mean_ms"] if t["mean_ms"] > 0 else 0
    print(f"\n{'─'*60}")
    print(f"  {model_name}  ({g['bytes_mb']:.1f} MB/cut, {DEVICE})")
    print(f"  gRPC:  {g['mean_ms']:7.2f} ± {g['std_ms']:5.2f} ms/batch  "
          f"({g['mean_gbps']:.2f} GB/s effective)")
    print(f"  TCP:   {t['mean_ms']:7.2f} ± {t['std_ms']:5.2f} ms/batch  "
          f"({t['mean_gbps']:.2f} GB/s effective)")
    print(f"  TCP speedup: {speedup:.2f}×")


if __name__ == "__main__":
    models = sys.argv[1:] or list(MODELS.keys())
    print(f"Transport benchmark  device={DEVICE}  repeats={REPEATS}  "
          f"warm-up=1 epoch  measured={EPOCHS-1} epochs")
    all_results = {}
    for name in models:
        if name not in MODELS:
            print(f"unknown model '{name}', choices: {list(MODELS.keys())}"); continue
        print(f"\n[{name}] running...", flush=True)
        res = bench(name, MODELS[name])
        all_results[name] = res
        print_results(name, res)

    print(f"\n{'═'*60}")
    print("Summary  (TCP speedup over gRPC):")
    for name, r in all_results.items():
        g = r.get("grpc", {}); t = r.get("tcp", {})
        if g and t:
            print(f"  {name:8s}  {r['grpc']['bytes_mb']:.1f} MB/cut  "
                  f"{g['mean_ms']:.1f}ms → {t['mean_ms']:.1f}ms  "
                  f"({g['mean_ms']/t['mean_ms']:.2f}×)")
