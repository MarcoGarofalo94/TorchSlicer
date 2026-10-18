"""Privacy suite benchmark: attack leakage vs defence, accuracy cost, mean ± std over seeds.

    .venv/bin/python benchmarks/privacy/run_privacy.py benchmarks/privacy/configs/label_mnist.yaml [--smoke]

A config names a dataset, a model, the cut, the attacks (``privacy.attacks`` entries) and a
list of ``runs``, each with its own defences: cut hooks (``sl.cut_hooks``) and/or
``privacy.defences`` (a run may also override ``cut`` and ``attacks``). Every run trains once per seed with the same data subset; the last
epoch's ``leak_*``, ``test_*`` and ``loss`` are aggregated. ``binary_positive: k`` turns the
labels into ``y == k`` (an imbalanced binary task, as in label-leakage papers).
Results: ``benchmarks/privacy/results/<config name>.json`` and a table on stdout.
``--smoke``: 1 seed, 1 epoch, 1/10 of the data. ``--seed k``: only seed ``k``, written to
``<name>.seed<k>.json``, so seeds can run as parallel processes; ``--merge`` then aggregates
those files into ``<name>.json``.

``dataset: shakespeare`` with ``model: gpt2`` fine-tunes pretrained GPT-2 (Hugging Face) on
``data/shakespeare/input.txt`` chunks of ``seq_len`` tokens (next-token prediction);
``train_subset``/``test_subset`` count sequences.
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
ROOT = os.path.dirname(os.path.dirname(HERE))


class _Binary(torch.utils.data.Dataset):
    def __init__(self, base, positive):
        self.base, self.positive = base, positive

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, int(y == self.positive)


def _gpt2():
    from transformers import GPT2LMHeadModel
    return GPT2LMHeadModel.from_pretrained("gpt2")


def _text(c, smoke):
    from transformers import GPT2TokenizerFast
    text = open(os.path.join(ROOT, c.get("root", "data"), "shakespeare", "input.txt")).read()
    ids = torch.tensor(GPT2TokenizerFast.from_pretrained("gpt2")(text)["input_ids"])
    T = c.get("seq_len", 32)
    seqs = ids[:(len(ids) - 1) // T * T + 1]
    x, y = seqs[:-1].reshape(-1, T), seqs[1:].reshape(-1, T)
    g = torch.Generator().manual_seed(c.get("data_seed", 0))
    perm = torch.randperm(x.shape[0], generator=g)
    n_tr, n_te = c["train_subset"], c["test_subset"]
    if smoke:
        n_tr, n_te = max(32, n_tr // 10), max(32, n_te // 10)
    tr, te = perm[:n_tr], perm[n_tr:n_tr + n_te]
    return (torch.utils.data.TensorDataset(x[tr], y[tr]), torch.utils.data.TensorDataset(x[te], y[te]))


def _model(name, in_ch, n_classes):
    if name == "cnn":           # conv(0) relu(1) pool(2) conv(3) relu(4) pool(5) flat(6) fc(7) relu(8) fc(9)
        return nn.Sequential(nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Flatten(), nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, n_classes))
    raise ValueError(f"unknown model {name!r}")


def _data(c, smoke):
    if c["dataset"] == "shakespeare":
        return _text(c, smoke)
    params = c.get("dataset_params", {})            # e.g. dataset: npz with path / test_path / mean / std
    tr = load_dataset(c["dataset"], root=c.get("root", "data"), train=True, **params)
    te = load_dataset(c["dataset"], root=c.get("root", "data"), train=False, **params)
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
    lm = c.get("model") == "gpt2"
    if lm:
        model = _gpt2()
    else:
        model = _model(c.get("model", "cnn"), x0.shape[0], n_classes)
        model(x0[None])                                 # materialize lazy layers
    model = model.to(dev)
    cfg = ts.RunConfig()
    cfg.logging.enabled = False
    cfg.training.epochs = 1 if smoke else c["epochs"]
    cfg.training.optimizer = c.get("optimizer", {"name": "Adam", "params": {"lr": 1e-3}})
    cfg.training.criterion = {"name": "CrossEntropyLoss", "params": {}}
    cfg.sl.cut_hooks = copy.deepcopy(run.get("cut_hooks", []))
    cfg.privacy.attacks = copy.deepcopy(run.get("attacks", c.get("attacks", [])))
    cfg.privacy.defences = copy.deepcopy(run.get("defences", []))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(tr, batch_size=c.get("batch_size", 64), shuffle=True, generator=gen)
    test = DataLoader(te, batch_size=256)
    if lm:
        sys.path.insert(0, ROOT)
        from examples.pack.gpt2 import pack_gpt2
        sliced = ts.slice(model, cuts=[str(run.get("cut", c["cut"]))], pack=pack_gpt2)
    else:
        sliced = ts.slice(model, cuts=[str(run.get("cut", c["cut"]))])
    t0 = time.perf_counter()
    h = sliced.train(loader, run_config=cfg, eval_data=test, eval_every=cfg.training.epochs)
    last = h[-1]
    out = {k: v for k, v in last.items() if k.startswith(("leak_", "test_")) or k == "loss"}
    out["time_s"] = round(time.perf_counter() - t0, 2)
    out["bytes_fwd"] = last.get("bytes_fwd", 0)
    return out


def _aggregate(label, seeds, per_seed):
    keys = [k for k in per_seed[0] if all(k in r for r in per_seed)]
    agg = {k: (st.mean(r[k] for r in per_seed), st.stdev(r[k] for r in per_seed) if len(per_seed) > 1 else 0.0)
           for k in keys if isinstance(per_seed[0][k], (int, float))}
    print(label, "  ".join(f"{k}={m:.4g}±{s:.2g}" for k, (m, s) in agg.items()
                           if k.startswith(("leak_", "test_loss", "test_acc", "loss")) and not k.endswith("batches")),
          flush=True)
    return {"run": label, "seeds": seeds, "mean_std": agg, "per_seed": per_seed}


def main():
    path = sys.argv[1]
    smoke = "--smoke" in sys.argv
    c = yaml.safe_load(open(path))
    name = os.path.splitext(os.path.basename(path))[0] + ("_smoke" if smoke else "")
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    if "--merge" in sys.argv:
        parts = sorted(f for f in os.listdir(out_dir) if f.startswith(f"{name}.seed") and f.endswith(".json"))
        loaded = [json.load(open(os.path.join(out_dir, f))) for f in parts]
        results = []
        for i, run in enumerate(c["runs"]):
            rows = [(p["seed"], p["results"][i]) for p in loaded]
            results.append(_aggregate(run["label"], [s for s, _ in rows], [r for _, r in rows]))
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump({"config": c, "results": results}, f, indent=1)
        return
    seeds = c.get("seeds", [0, 1, 2])[:1 if smoke else None]
    if "--seed" in sys.argv:
        seeds = [int(sys.argv[sys.argv.index("--seed") + 1])]
    tr, te = _data(c, smoke)
    if "--seed" in sys.argv:
        rows = [run_one(c, run, seeds[0], tr, te, smoke) for run in c["runs"]]
        with open(os.path.join(out_dir, f"{name}.seed{seeds[0]}.json"), "w") as f:
            json.dump({"seed": seeds[0], "results": rows}, f, indent=1)
        return
    results = [_aggregate(run["label"], seeds, [run_one(c, run, s, tr, te, smoke) for s in seeds])
               for run in c["runs"]]
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump({"config": c, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
