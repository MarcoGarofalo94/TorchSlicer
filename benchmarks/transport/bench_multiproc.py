"""Multi-process tensor transport benchmark.

Launches a real follower worker as a subprocess (separate CUDA context) so that
CUDA IPC and NCCL can be tested in addition to gRPC and TCP.

Usage:
    .venv/bin/python benchmarks/transport/bench_multiproc.py [model ...]
    .venv/bin/python benchmarks/transport/bench_multiproc.py --transports grpc,tcp,ipc,nccl
    .venv/bin/python benchmarks/transport/bench_multiproc.py medium large

Models: small medium large xlarge  (same as bench_transport.py)
"""
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib" / "torchslicer"))
import torchslicer as ts
from torchslicer.config import RunConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS  = 4    # first epoch warm-up
REPEATS = 3
PYTHON  = sys.executable


# ── model definitions (same as bench_transport.py) ────────────────────────────

def _small():
    return nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 10),
    ), (64, 512), 10

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
        nn.Linear(1024, 2048), nn.GELU(),
        nn.Linear(2048, 2048), nn.GELU(),
        nn.Linear(2048, 2048), nn.GELU(),
        nn.Linear(2048, 2048), nn.GELU(),
        nn.Linear(2048, 10),
    ), (64, 1024), 10

def _xlarge():
    d, seq, heads = 768, 256, 12
    return nn.Sequential(
        nn.TransformerEncoderLayer(d, heads, dim_feedforward=d*4, batch_first=True),
        nn.TransformerEncoderLayer(d, heads, dim_feedforward=d*4, batch_first=True),
        nn.Flatten(1),
        nn.Linear(d * seq, 10),
    ), (4, seq, d), 10

MODELS = {"small": _small, "medium": _medium, "large": _large, "xlarge": _xlarge}


# ── helpers ───────────────────────────────────────────────────────────────────

def _free_pair():
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

def _loader(input_shape, n_classes, n_batches=16):
    B = input_shape[0]
    g = torch.Generator().manual_seed(42)
    x = torch.randn(*input_shape, generator=g)
    y = torch.randint(0, n_classes, (B,), generator=g)
    return [(x, y)] * n_batches

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


# ── subprocess worker ─────────────────────────────────────────────────────────

# When launched as a subprocess worker, this module runs the block below.
_WORKER_SENTINEL = "--_worker_mode"

def _run_as_worker():
    """Entry point when this script is invoked as a worker subprocess."""
    port = int(os.environ["_BENCH_WORKER_PORT"])
    transport = os.environ["_BENCH_TRANSPORT"]
    env_extra = {}
    if os.environ.get("TCP_NO_PINNED"):
        env_extra["TCP_NO_PINNED"] = "1"
    # Silence coordinator logs from the worker side.
    os.environ.setdefault("TORCHSLICER_LOG_LEVEL", "WARNING")
    cfg = _cfg(transport)
    # P2P follower: no coordinator_addr → waits for init() from driver.
    ts.run_worker(port=port, coordinator_addr=None, device=DEVICE)


def _start_worker_proc(port: int, transport: str, extra_env: dict | None = None) -> subprocess.Popen:
    env = {**os.environ,
           "_BENCH_WORKER_PORT": str(port),
           "_BENCH_TRANSPORT": transport,
           "TENSOR_TRANSPORT": transport,
           "DEVICE": DEVICE,
           "TORCHSLICER_LOG_LEVEL": "WARNING",
           **(extra_env or {})}
    proc = subprocess.Popen(
        [PYTHON, __file__, _WORKER_SENTINEL],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


# ── benchmark core ────────────────────────────────────────────────────────────

def _run_once(model_fn, transport: str, extra_env: dict | None = None) -> dict:
    model, input_shape, n_classes = model_fn()
    loader  = _loader(input_shape, n_classes)
    cfg     = _cfg(transport)

    follower_port = _free_pair()
    driver_port   = _free_pair()
    follower_addr = f"127.0.0.1:{follower_port}"
    driver_addr   = f"127.0.0.1:{driver_port}"

    proc = _start_worker_proc(follower_port, transport, extra_env)
    try:
        # Poll until the worker's gRPC port is accepting connections (max 8 s).
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"worker subprocess exited early (rc={proc.returncode})")
            try:
                with socket.create_connection(("127.0.0.1", follower_port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            raise RuntimeError(f"worker subprocess did not open port {follower_port} within 8 s")

        executor = ts.P2PExecutor(
            peers=[driver_addr, follower_addr],
            address=driver_addr,
            bind_addr=driver_addr,
            run_config=cfg,
        )
        sliced = ts.slice(model, n=2, executor=executor)
        t0 = time.perf_counter()
        history = sliced.train(loader, run_config=cfg)
        total_s = time.perf_counter() - t0
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    n_epochs = len(history)
    batches_per_epoch = len(loader)
    bytes_total_epoch = history[-1].get("bytes_total", 0)
    ms_per_batch = (total_s / n_epochs) / batches_per_epoch * 1e3
    epoch_s      = total_s / n_epochs
    gbps = (bytes_total_epoch / epoch_s) / 1e9 if bytes_total_epoch and epoch_s else 0.0
    bytes_fwd_list = history[-1].get("bytes_fwd", [0])
    bytes_fwd = bytes_fwd_list[0] if isinstance(bytes_fwd_list, list) else bytes_fwd_list
    return {"ms_per_batch": ms_per_batch, "gbps": gbps,
            "bytes_mb": bytes_fwd / 1e6, "total_s": total_s}


def _supported_transports() -> list[str]:
    supported = ["grpc", "tcp"]
    if torch.cuda.is_available():
        supported.append("ipc")
        try:
            import torch.distributed as _dist  # noqa: F401
            supported.append("nccl")
        except ImportError:
            pass
    return supported


def bench(model_name: str, model_fn, transports=None) -> dict:
    if transports is None:
        transports = _supported_transports()
    results = {}
    for transport in transports:
        runs = []
        for r in range(REPEATS):
            try:
                runs.append(_run_once(model_fn, transport))
            except Exception as e:
                print(f"  ERROR {transport} run {r}: {e}", flush=True)
        if runs:
            ms   = [x["ms_per_batch"] for x in runs]
            gbps = [x["gbps"]         for x in runs]
            results[transport] = {
                "mean_ms":   statistics.mean(ms),
                "std_ms":    statistics.stdev(ms) if len(ms) > 1 else 0.0,
                "mean_gbps": statistics.mean(gbps),
                "bytes_mb":  runs[0]["bytes_mb"],
            }
    return results


def print_results(model_name: str, r: dict):
    g = r.get("grpc", {})
    if not g:
        # No gRPC baseline — use first available as reference.
        first_key = next(iter(r), None)
        if not first_key:
            print(f"  {model_name}: no results"); return
        g = r[first_key]
    print(f"\n{'─'*60}")
    print(f"  {model_name}  ({g['bytes_mb']:.1f} MB/cut, {DEVICE})")
    ref_ms = g["mean_ms"]
    for label, key in [("gRPC", "grpc"), ("TCP ", "tcp"), ("IPC ", "ipc"), ("NCCL", "nccl")]:
        t = r.get(key)
        if not t:
            continue
        speedup = ref_ms / t["mean_ms"] if t["mean_ms"] > 0 else 0
        suffix  = f"  {speedup:.2f}× over gRPC" if key != "grpc" else ""
        print(f"  {label}: {t['mean_ms']:7.2f} ± {t['std_ms']:5.2f} ms/batch"
              f"  ({t['mean_gbps']:.2f} GB/s){suffix}")


if __name__ == "__main__":
    if _WORKER_SENTINEL in sys.argv:
        _run_as_worker()
        sys.exit(0)

    args = sys.argv[1:]
    transports = None
    if "--transports" in args:
        idx = args.index("--transports")
        transports = args[idx + 1].split(",")
        args = args[:idx] + args[idx + 2:]
    if transports is None:
        transports = _supported_transports()

    models = args or list(MODELS.keys())
    print(f"Multi-process transport benchmark  device={DEVICE}  repeats={REPEATS}"
          f"  epochs={EPOCHS} (warm-up=1)  transports={','.join(transports)}")
    print("(Each follower worker runs as a real subprocess — separate CUDA context)")

    all_results = {}
    for name in models:
        if name not in MODELS:
            print(f"unknown model '{name}', choices: {list(MODELS.keys())}"); continue
        print(f"\n[{name}] running...", flush=True)
        res = bench(name, MODELS[name], transports=transports)
        all_results[name] = res
        print_results(name, res)

    print(f"\n{'═'*60}")
    print("Summary  (speedup over gRPC):")
    for name, r in all_results.items():
        g = r.get("grpc", {})
        if not g:
            continue
        parts = [f"{name:8s}  {g['bytes_mb']:.1f} MB/cut  gRPC {g['mean_ms']:.1f}ms"]
        for key in ("tcp", "ipc", "nccl"):
            t = r.get(key)
            if t:
                parts.append(f"{key.upper()} {t['mean_ms']:.1f}ms ({g['mean_ms']/t['mean_ms']:.2f}×)")
        print("  " + "  ".join(parts))
