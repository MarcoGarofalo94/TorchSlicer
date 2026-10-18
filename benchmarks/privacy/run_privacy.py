"""Privacy suite benchmark: attack leakage vs defence, accuracy cost, mean ± std over seeds.

    .venv/bin/python benchmarks/privacy/run_privacy.py benchmarks/privacy/configs/label_mnist.yaml [--smoke]

A config names a dataset, a model, the cut, the attacks (``privacy.attacks`` entries) and a
list of ``runs``, each with its own defences: cut hooks (``sl.cut_hooks``) and/or
``privacy.defences``. Every run trains once per seed with the same data subset; the last
epoch's ``leak_*``, ``test_*`` and ``loss`` are aggregated. ``binary_positive: k`` turns the
labels into ``y == k`` (an imbalanced binary task, as in label-leakage papers).
Results: ``benchmarks/privacy/results/<config name>.json`` and a table on stdout.
``--smoke``: 1 seed, 1 epoch, 1/10 of the data.
"""
import copy
import json
import os
import statistics as st
import sys
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

import torchslicer as ts
from torchslicer.data import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))


class _Binary(torch.utils.data.Dataset):
    def __init__(self, base, positive):
        self.base, self.positive = base, positive

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, int(y == self.positive)


def _model(name, in_ch, n_classes):
    if name == "cnn":           # conv(0) relu(1) pool(2) conv(3) relu(4) pool(5) flat(6) fc(7) relu(8) fc(9)
        return nn.Sequential(nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Flatten(), nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, n_classes))
    raise ValueError(f"unknown model {name!r}")


def _data(c, smoke):
    tr = load_dataset(c["dataset"], root=c.get("root", "data"), train=True)
    te = load_dataset(c["dataset"], root=c.get("root", "data"), train=False)
    g = torch.Generator().manual_seed(c.get("data_seed", 0))
    n_tr = c.get("train_subset", 0) or len(tr)
    n_te = c.get("test_subset", 0) or len(te)
    if smoke:
        n_tr, n_te = max(256, n_tr // 10), max(256, n_te // 10)
    tr = Subset(tr, torch.randperm(len(tr), generator=g)[:n_tr].tolist())
    te = Subset(te, torch.randperm(len(te), generator=g)[:n_te].tolist())
    if "binary_positive" in c:
        tr, te = _Binary(tr, c["binary_positive"]), _Binary(te, c["binary_positive"])
    return tr, te


def run_one(c, run, seed, tr, te, smoke):
    torch.manual_seed(seed)
    n_classes = 2 if "binary_positive" in c else c.get("classes", 10)
    x0 = tr[0][0]
    dev = torch.device(c.get("device", "cpu") if torch.cuda.is_available() else "cpu")
    model = _model(c.get("model", "cnn"), x0.shape[0], n_classes)
    model(x0[None])                                     # materialize lazy layers
    model = model.to(dev)
    cfg = ts.RunConfig()
    cfg.logging.enabled = False
    cfg.training.epochs = 1 if smoke else c["epochs"]
    cfg.training.optimizer = c.get("optimizer", {"name": "Adam", "params": {"lr": 1e-3}})
    cfg.training.criterion = {"name": "CrossEntropyLoss", "params": {}}
    cfg.sl.cut_hooks = copy.deepcopy(run.get("cut_hooks", []))
    cfg.privacy.attacks = copy.deepcopy(c.get("attacks", []))
    cfg.privacy.defences = copy.deepcopy(run.get("defences", []))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(tr, batch_size=c.get("batch_size", 64), shuffle=True, generator=gen)
    test = DataLoader(te, batch_size=256)
    sliced = ts.slice(model, cuts=[str(run.get("cut", c["cut"]))])
    t0 = time.perf_counter()
    h = sliced.train(loader, run_config=cfg, eval_data=test, eval_every=cfg.training.epochs)
    last = h[-1]
    out = {k: v for k, v in last.items() if k.startswith(("leak_", "test_")) or k == "loss"}
    out["time_s"] = round(time.perf_counter() - t0, 2)
    out["bytes_fwd"] = last.get("bytes_fwd", 0)
    return out


def main():
    path = sys.argv[1]
    smoke = "--smoke" in sys.argv
    c = yaml.safe_load(open(path))
    seeds = c.get("seeds", [0, 1, 2])[:1 if smoke else None]
    tr, te = _data(c, smoke)
    results = []
    for run in c["runs"]:
        per_seed = [run_one(c, run, s, tr, te, smoke) for s in seeds]
        keys = [k for k in per_seed[0] if all(k in r for r in per_seed)]
        agg = {k: (st.mean(r[k] for r in per_seed), st.stdev(r[k] for r in per_seed) if len(per_seed) > 1 else 0.0)
               for k in keys if isinstance(per_seed[0][k], (int, float))}
        results.append({"run": run["label"], "seeds": seeds, "mean_std": agg, "per_seed": per_seed})
        print(run["label"], "  ".join(f"{k}={m:.4g}±{s:.2g}" for k, (m, s) in agg.items()
                                      if k.startswith(("leak_", "test_loss", "test_acc", "loss")) and not k.endswith("batches")),
              flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    name = os.path.splitext(os.path.basename(path))[0] + ("_smoke" if smoke else "")
    with open(os.path.join(HERE, "results", f"{name}.json"), "w") as f:
        json.dump({"config": c, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
