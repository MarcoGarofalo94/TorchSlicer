"""Tensor transport benchmark: grpc / tcp (pinned or not) / ipc / nccl, real OS processes.

A P2P run with 2 slices: the driver (this process) holds slice 0, the follower is a
subprocess with its own CUDA context, so IPC and NCCL are real. Per run it reports:
  ms/batch      steady state: mean epoch duration over epochs 2.. (epoch 1 = warm-up),
                divided by batches; setup (discovery, probing, init) is excluded
  compute       ms/batch spent in forward/backward/step on both slices (CUDA-synced)
  other         ms/batch - compute: transfers, serialization, RPCs, scheduling
  cut MB        bytes of ONE activation crossing the cut per batch (the gradient back
                is the same size)

Usage:
    .venv/bin/python benchmarks/transport/bench_transport.py [model ...] [--variants grpc,tcp,...]
    BENCH_REPEATS=3 BENCH_EPOCHS=4 BENCH_THREADS=4 (default: the cgroup CPU quota)

Models: small medium large xlarge (2-slice splits of small nets, cuts of 0.07-3 MB) and
act8 act32 act64 (an MLP whose cut is 8/32/64 MB per batch).
Variants: grpc tcp tcp-nopin (TCP without the pinned buffer) ipc nccl (needs one GPU
per slice, skipped on fewer than 2 GPUs). Results: results/<device>.json.

Threads: torch sizes its CPU thread pool from the host's cores, not the container's
CPU quota. Oversubscribed, every CPU tensor op on the receive path spins idle threads
that burn the quota (measured: gRPC 110 -> 22 ms/batch on 'medium' once matched), so
both processes get BENCH_THREADS threads.
"""
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lib" / "torchslicer"))
import torchslicer as ts
import torchslicer.executors.worker as _worker
from torchslicer.config import RunConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = int(os.environ.get("BENCH_EPOCHS", 4))
REPEATS = int(os.environ.get("BENCH_REPEATS", 3))
N_BATCHES = 16
_WORKER_FLAG = "--_worker_mode"


def _cpu_quota() -> int:
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


THREADS = int(os.environ.get("BENCH_THREADS", _cpu_quota()))


# ── models: (module, input shape, n classes) ──────────────────────────────────

def _small():
    return nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 10)), (64, 512), 10

def _medium():
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(8),
        nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
        nn.Flatten(),
        nn.Linear(256 * 4 * 4, 10),
    ), (64, 3, 32, 32), 10

def _large():
    return nn.Sequential(
        nn.Linear(1024, 2048), nn.GELU(), nn.Linear(2048, 2048), nn.GELU(),
        nn.Linear(2048, 2048), nn.GELU(), nn.Linear(2048, 2048), nn.GELU(),
        nn.Linear(2048, 10),
    ), (64, 1024), 10

def _xlarge():
    d, seq, heads = 768, 256, 12
    return nn.Sequential(
        nn.TransformerEncoderLayer(d, heads, dim_feedforward=d * 4, batch_first=True),
        nn.TransformerEncoderLayer(d, heads, dim_feedforward=d * 4, batch_first=True),
        nn.Flatten(1),
        nn.Linear(d * seq, 10),
    ), (4, seq, d), 10

def _act(mb):
    """An MLP split after its 3rd layer: the cut is (batch, 1024) fp32 = ``mb`` MB."""
    def build():
        d = 1024
        return nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(),
                             nn.Linear(d, 10)), (mb * 2**20 // (4 * d), d), 10
    return build

MODELS = {"small": _small, "medium": _medium, "large": _large, "xlarge": _xlarge,
          "act8": _act(8), "act32": _act(32), "act64": _act(64)}
VARIANTS = ["grpc", "tcp", "tcp-nopin", "ipc", "nccl"]


def _transport(variant: str) -> str:
    return "tcp" if variant == "tcp-nopin" else variant


def _cfg(variant: str) -> RunConfig:
    cfg = RunConfig()
    cfg.training.optimizer = {"name": "SGD", "params": {"lr": 0.01, "momentum": 0.9}}
    cfg.training.criterion = {"name": "CrossEntropyLoss", "params": {}}
    cfg.training.epochs = EPOCHS
    cfg.logging.enabled = False
    cfg.network.device = DEVICE
    cfg.transport.tensor = _transport(variant)
    cfg.sl.setting = "vanilla"
    return cfg


def _free_pair() -> int:
    """A port p with p+1 free too (the TCP tensor listener)."""
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


def _loader(input_shape, n_classes, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*input_shape, generator=g)
    y = torch.randint(0, n_classes, (input_shape[0],), generator=g)
    return [(x, y)] * N_BATCHES


class _Epochs(ts.TrainingCallback):
    def __init__(self):
        self.metrics = []

    def on_epoch_end(self, epoch, metrics):
        self.metrics.append(metrics)
        return metrics


def _run_once(model_fn, variant: str, seed: int) -> dict:
    torch.manual_seed(seed)
    model, input_shape, n_classes = model_fn()
    loader = _loader(input_shape, n_classes, seed)
    cfg = _cfg(variant)
    nopin = variant == "tcp-nopin"
    _worker._TCP_NO_PINNED = nopin            # the driver's sends (activations)
    follower_port, driver_port = _free_pair(), _free_pair()
    env = {**os.environ, "_BENCH_WORKER_PORT": str(follower_port),
           "TENSOR_TRANSPORT": cfg.transport.tensor, "DEVICE": DEVICE,
           "OMP_NUM_THREADS": str(THREADS), "TCP_NO_PINNED": "1" if nopin else "0"}
    proc = subprocess.Popen([sys.executable, __file__, _WORKER_FLAG], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 30
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"worker subprocess exited early (rc={proc.returncode})")
            try:
                with socket.create_connection(("127.0.0.1", follower_port), timeout=0.1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("worker subprocess did not open its port")
                time.sleep(0.05)
        driver = f"127.0.0.1:{driver_port}"
        executor = ts.P2PExecutor(peers=[driver, f"127.0.0.1:{follower_port}"],
                                  address=driver, bind_addr=driver, run_config=cfg)
        epochs = _Epochs()
        ts.slice(model, n=2, executor=executor).train(loader, run_config=cfg, callbacks=[epochs])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    steady = epochs.metrics[1:] or epochs.metrics
    ms = statistics.mean(m["duration_s"] for m in steady) * 1e3 / N_BATCHES
    compute = statistics.mean(sum(m["compute_s"]) for m in steady) * 1e3 / N_BATCHES
    fwd = steady[-1]["bytes_fwd"]
    cut_mb = (fwd[0] if isinstance(fwd, list) else fwd) / N_BATCHES / 1e6
    return {"ms": ms, "compute_ms": compute, "cut_mb": cut_mb}


def _skip_reason(variant: str):
    if variant in ("ipc", "nccl") and DEVICE != "cuda":
        return "needs CUDA"
    if variant == "nccl" and torch.cuda.device_count() < 2:
        return "needs one GPU per slice (NCCL refuses two ranks on one GPU)"
    return None


def bench(model_name: str, variants) -> dict:
    out = {}
    for variant in variants:
        reason = _skip_reason(variant)
        if reason:
            out[variant] = {"skipped": reason}
            continue
        runs = []
        for seed in range(REPEATS):
            try:
                runs.append(_run_once(MODELS[model_name], variant, seed))
            except Exception as e:
                print(f"  ERROR {model_name} {variant} seed {seed}: {e}", flush=True)
        if runs:
            col = lambda k: [r[k] for r in runs]
            sd = lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0
            out[variant] = {"ms": statistics.mean(col("ms")), "ms_std": sd(col("ms")),
                            "compute_ms": statistics.mean(col("compute_ms")),
                            "cut_mb": runs[0]["cut_mb"], "n": len(runs)}
    return out


def print_table(results: dict):
    print(f"\n{'model':7s} {'cut MB':>7s}  {'variant':9s} {'ms/batch':>16s} {'compute':>8s} "
          f"{'other':>8s} {'vs grpc':>8s} {'vs tcp':>7s}")
    for name, r in results.items():
        for variant, v in r.items():
            if "skipped" in v:
                print(f"{name:7s} {'':>7s}  {variant:9s} skipped: {v['skipped']}")
                continue
            rel = lambda ref: f"{r[ref]['ms'] / v['ms']:.2f}x" if "ms" in r.get(ref, {}) else "-"
            print(f"{name:7s} {v['cut_mb']:7.2f}  {variant:9s} {v['ms']:8.2f} ± {v['ms_std']:5.2f} "
                  f"{v['compute_ms']:8.2f} {v['ms'] - v['compute_ms']:8.2f} {rel('grpc'):>8s} "
                  f"{rel('tcp'):>7s}")


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        ts.run_worker(port=int(os.environ["_BENCH_WORKER_PORT"]), coordinator_addr=None,
                      device=DEVICE)
        sys.exit(0)
    torch.set_num_threads(THREADS)
    args = sys.argv[1:]
    variants = VARIANTS
    if "--variants" in args:
        i = args.index("--variants")
        variants = args[i + 1].split(",")
        args = args[:i] + args[i + 2:]
    models = args or list(MODELS)
    print(f"transport benchmark device={DEVICE} gpus={torch.cuda.device_count()} "
          f"threads={THREADS} repeats={REPEATS} epochs={EPOCHS} (warm-up 1) batches={N_BATCHES}",
          flush=True)
    results = {}
    for name in models:
        print(f"[{name}] ...", flush=True)
        results[name] = bench(name, variants)
    print_table(results)
    (HERE / "results").mkdir(exist_ok=True)
    meta = {"device": DEVICE, "gpus": torch.cuda.device_count(), "threads": THREADS,
            "repeats": REPEATS, "epochs": EPOCHS, "batches": N_BATCHES}
    path = HERE / "results" / f"{DEVICE}.json"
    path.write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    print(f"\nwrote {path}")
