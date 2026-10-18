"""Cost of NAT traversal: the same U-shaped P2P run (one OS process per node, loopback)
with every node reachable, with the driver behind NAT (its reverse traffic comes back
over its tunnel), and with every node behind NAT through a relay process.

usage: python benchmarks/nat/bench_nat.py [repeats]    -> results/bench_nat.json
Per batch the cuts carry 4 x 1 MiB (activations to and from the follower, gradients
back); the driver receives 2 MiB of them. Time per batch = mean epoch time over the
epochs after the first (warm-up), divided by the batches.
"""
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

from torchslicer.config import RunConfig
from torchslicer.deploy.launch import run_processes

HERE = Path(__file__).resolve().parent
MODES = {
    "direct-grpc": {"TENSOR_TRANSPORT": "grpc"},
    "direct-tcp":  {"TENSOR_TRANSPORT": "tcp"},
    "nat-grpc":    {"TENSOR_TRANSPORT": "grpc", "TRANSPORT_NAT": "true"},
    "nat-tcp":     {"TENSOR_TRANSPORT": "tcp", "TRANSPORT_NAT": "true"},
    "relay":       {"TENSOR_TRANSPORT": "grpc", "TRANSPORT_NAT": "true", "TRANSPORT_RELAY": "auto"},
}


def one(env: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "history.json")
        full = {**env, "BENCH_OUT": out, "OMP_NUM_THREADS": "1", "LOG_LEVEL": "WARNING"}
        old = {k: os.environ.get(k) for k in full}
        os.environ.update(full)
        try:
            cfg = RunConfig.load(str(HERE / "bench.yaml"))
            code = run_processes(str(HERE / "train.py"), str(HERE / "bench.yaml"), cfg,
                                 out=os.path.join(tmp, "logs"), timeout=600)
        finally:
            for k, v in old.items():
                os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)
        assert code == 0, code
        h = json.load(open(out))
    n = int(os.environ.get("BENCH_BATCHES", 32))
    return {"s_per_batch": statistics.mean(e["epoch_s"] for e in h[1:]) / n,
            "losses": [e["loss"] for e in h], "bytes_total": h[-1]["bytes_total"]}


if __name__ == "__main__":
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    results = {}
    for mode, env in MODES.items():
        runs = [one(env) for _ in range(repeats)]
        t = [r["s_per_batch"] * 1e3 for r in runs]
        results[mode] = {"ms_per_batch": t, "mean": statistics.mean(t),
                         "std": statistics.stdev(t) if len(t) > 1 else 0.0,
                         "losses": runs[0]["losses"], "bytes_total": runs[0]["bytes_total"]}
        print(f"{mode:12s} {results[mode]['mean']:7.1f} ± {results[mode]['std']:5.1f} ms/batch  "
              f"losses {runs[0]['losses']}", flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "bench_nat.json").write_text(json.dumps(results, indent=1))
