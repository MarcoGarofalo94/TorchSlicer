"""Every baseline in baselines/ runs: each of its runs trains one epoch on synthetic data
of its dataset's shape (CI has no datasets), reports its metric, and its config is valid.
The published numbers are checked by ``baselines/run.py <name> --check``, not here."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "baselines"))
import run as baselines  # noqa: E402

NAMES = sorted(d for d in os.listdir(os.path.join(ROOT, "baselines"))
               if os.path.isfile(os.path.join(ROOT, "baselines", d, "config.yaml")))


def test_there_are_baselines():
    assert NAMES


@pytest.mark.parametrize("name", NAMES)
def test_baseline_runs_on_synthetic_data(name):
    spec, setting = baselines.load(name)
    for key in ("paper", "model", "metric", "runs", "synthetic", "smoke"):
        assert key in spec, f"{name}: baseline.{key} missing"
    assert spec["paper"].get("url"), f"{name}: the paper needs a link"
    syn = spec["synthetic"]
    if "data" not in syn:      # shorthand: the synthetic image dataset with this shape/classes
        syn = {"data": {"dataset": "synthetic", "params": {**syn, "n": 192},
                        "train_subset": 0, "test_subset": 0, "batch_size": 32}}
    spec = {**spec, "smoke": baselines._merge(spec["smoke"], {"data": syn["data"]}),
            "model": syn.get("model", spec["model"])}
    for run in spec["runs"]:
        assert "expected" in run, f"{name}/{run['label']}: expected missing (null = not checked)"
        assert run["expected"] is None or {"value", "tol"} <= set(run["expected"]), f"{name}/{run['label']}"
        if "cuts" in syn:
            run = {**run, "cuts": syn["cuts"]}
        out = baselines.run_one(spec, setting, run, seed=0, smoke=True)
        assert out["value"] is None or 0.0 <= out["value"] <= 1.0


def test_slperf_partitioner_matches_its_definition():
    """SLPerf's hetero split: per-class Dirichlet shares, then every client cut to the
    smallest client's size (no duplicates when the first attempt succeeds)."""
    import numpy as np
    import torchslicer as ts
    baselines.load("slperf_mnist")                        # registers slperf_hetero
    labels = np.repeat(np.arange(10), 300)
    split = ts.data.partition(labels, 3, "slperf_hetero", seed=0, alpha=0.1)
    sizes = [len(i) for i in split.indices]
    assert len(set(sizes)) == 1 and sizes[0] < len(labels) // 3
    assert len(np.unique(np.concatenate(split.indices))) == sum(sizes)
    again = ts.data.partition(labels, 3, "slperf_hetero", seed=0, alpha=0.1)
    assert all((a == b).all() for a, b in zip(split.indices, again.indices))
