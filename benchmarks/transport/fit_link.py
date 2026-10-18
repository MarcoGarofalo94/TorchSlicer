"""Fit the simulator's link model (``sl.link_latency_ms``, ``sl.link_bandwidth_mbps``) to a
measured transport benchmark, and report how well it predicts.

    .venv/bin/python benchmarks/transport/fit_link.py [results/cuda.json] [--max-mb 10]

Per batch, a 2-slice vanilla run sends 3 messages (activation, gradient, labels) and 2 cut
tensors. The benchmark's non-compute time per batch ("other") is fitted as
    other = 3 * latency + 2 * cut_bytes / bandwidth
by least squares over the models, per transport. The simulator computes the same formula
from its byte and message counts, so these are the values that make ``sim_comm_s``
reproduce that transport on that machine.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def fit(results: dict, variant: str, max_mb: float):
    pts = [(x["cut_mb"], x["ms"] - x["compute_ms"]) for r in results.values()
           for v, x in r.items() if v == variant and "ms" in x and x["cut_mb"] <= max_mb]
    if len(pts) < 2:
        return None
    a = np.array([[3.0, 2 * mb] for mb, _ in pts])
    y = np.array([o for _, o in pts])
    (lat_ms, ms_per_mb), *_ = np.linalg.lstsq(a, y, rcond=None)
    err = np.abs(a @ [lat_ms, ms_per_mb] - y) / y
    return {"latency_ms": round(float(lat_ms), 3),
            "bandwidth_mbps": round(8 / float(ms_per_mb) * 1e3, 1),      # MB/ms -> Mbit/s
            "n": len(pts), "median_rel_err": round(float(np.median(err)), 3),
            "max_rel_err": round(float(err.max()), 3)}


if __name__ == "__main__":
    args = sys.argv[1:]
    max_mb = 10.0
    if "--max-mb" in args:
        i = args.index("--max-mb")
        max_mb = float(args[i + 1])
        args = args[:i] + args[i + 2:]
    path = Path(args[0]) if args else HERE / "results" / "cuda.json"
    data = json.loads(path.read_text())
    print(f"{path.name}: {data['meta']}, cuts <= {max_mb} MB")
    out = {}
    for v in ("grpc", "tcp", "tcp-nopin", "ipc", "nccl"):
        f = fit(data["results"], v, max_mb)
        if f:
            out[v] = f
            print(f"  {v:9s} sl.link_latency_ms: {f['latency_ms']:6.2f}  sl.link_bandwidth_mbps: "
                  f"{f['bandwidth_mbps']:9.0f}   (n={f['n']}, error median {f['median_rel_err']:.0%}, "
                  f"max {f['max_rel_err']:.0%})")
    (HERE / "results" / f"link_fit_{path.stem}_{max_mb:g}mb.json").write_text(json.dumps(
        {"source": path.name, "max_mb": max_mb, "fits": out}, indent=1))
