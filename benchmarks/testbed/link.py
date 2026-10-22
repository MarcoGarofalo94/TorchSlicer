"""One bench_transport model trained across two real hosts; prints a RESULT line.

The models and batches are benchmarks/transport/bench_transport.py's (BENCH_MODEL, default
act8), so the numbers compare with its one-host results. Launch with
benchmarks/testbed/link_p2p.yaml on any target, e.g. baremetal:

    python -m torchslicer launch benchmarks/testbed/link.py -c benchmarks/testbed/link_p2p.yaml \\
        --target baremetal --hosts A,B --workdir /path/to/TorchSlicer

RESULT: steady ms/batch (epochs 2..), compute ms/batch (both slices), cut MB per batch.
"""
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transport"))
import bench_transport as bt   # noqa: E402

import torch  # noqa: E402
import torchslicer as ts  # noqa: E402

MODEL = os.environ.get("BENCH_MODEL", "act8")
SEED = int(os.environ.get("BENCH_SEED", 0))


def build():
    torch.manual_seed(SEED)
    return bt.MODELS[MODEL]()[0]


def data():
    torch.manual_seed(SEED)
    _, shape, classes = bt.MODELS[MODEL]()
    return bt._loader(shape, classes, SEED)


if __name__ == "__main__":
    epochs = bt._Epochs()
    history = ts.run(build, data, callbacks=[epochs])
    if history:
        steady = epochs.metrics[1:] or epochs.metrics
        n = bt.N_BATCHES
        fwd = steady[-1]["bytes_fwd"]
        print("RESULT " + json.dumps({
            "model": MODEL, "seed": SEED,
            "transport": os.environ.get("TENSOR_TRANSPORT", "grpc"),
            "ms": statistics.mean(h["duration_s"] for h in steady) * 1e3 / n,
            "ms_epochs": [h["duration_s"] * 1e3 / n for h in steady],
            "compute_ms": statistics.mean(sum(h["compute_s"]) for h in steady) * 1e3 / n,
            "cut_mb": (fwd[0] if isinstance(fwd, list) else fwd) / n / 1e6,
            "loss": [h["loss"] for h in history]}), flush=True)
