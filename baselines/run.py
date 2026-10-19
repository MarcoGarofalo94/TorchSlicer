"""Run a baseline: a published split-learning result, its setting and expected number.

    .venv/bin/python baselines/run.py slperf_mnist [--smoke] [--runs a,b] [--seed k] [--merge] [--check]

``baselines/<name>/config.yaml`` is a RunConfig (``training``, ``data``, ``sl``, ...: the
setting shared by its runs) plus a ``baseline`` section:

    baseline:
      paper: {title: ..., url: ..., table: ...}
      model: {name: lenet, params: {}}        # baselines/models.py
      metric: test_accuracy                   # from the history entries
      report: last                            # last | best (over the evaluated epochs)
      seeds: [0, 1, 2]
      smoke: {training: {epochs: 1}, data: {train_subset: 600, test_subset: 200}}
      synthetic: {shape: [1, 28, 28], classes: 10}   # the CI smoke test's stand-in data
      runs:
        - label: vanilla_iid
          config: {sl: {...}, data: {...}}    # merged over the shared setting (an
                                              # optimizer/criterion/params entry is replaced)
          cuts: ["2"]
          expected: {value: 0.9792, tol: 0.015}   # null: reported, not checked (no paper number)
          # metric: ...                           # per-run override of baseline.metric

Every run trains once per seed (``data.seed`` and the model's init follow the seed) and
reports ``metric`` (mean ± std). Results: ``baselines/<name>/results/<name>.json``.
``--seed k`` writes ``<name>.seed<k>.json`` only (run seeds as parallel processes) and
``--merge`` combines them (with ``--runs``, one file per seed and group of runs). ``--check`` exits non-zero if a merged run's mean is outside
``value ± tol`` and prints the table. ``--smoke`` applies ``smoke`` and one seed.
"""
import copy
import json
import os
import statistics as st
import sys
import time

import torch
import yaml

import torchslicer as ts

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from models import MODELS  # noqa: E402


_ATOMIC = {"optimizer", "criterion", "params", "partition_params"}   # replaced, not merged


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        nested = isinstance(v, dict) and isinstance(out.get(k), dict) and k not in _ATOMIC
        out[k] = _merge(out[k], v) if nested else copy.deepcopy(v)
    return out


_EVALUATE = {}   # baseline name -> extras.evaluate


def load(name: str) -> tuple[dict, dict]:
    """``(baseline section, RunConfig dict)``; imports ``<name>/extras.py`` if there is one
    (a baseline's own dataset, partitioner, model, hook or algorithm registrations). An
    ``evaluate(model, cfg, run, smoke) -> {metric: value}`` there runs after training, and its
    metrics can be the baseline's ``metric``."""
    extras = os.path.join(HERE, name, "extras.py")
    raw = yaml.safe_load(open(os.path.join(HERE, name, "config.yaml")))
    spec = raw.pop("baseline")
    if os.path.isfile(extras):
        import importlib.util
        loc = importlib.util.spec_from_file_location(f"baseline_{name}_extras", extras)
        mod = importlib.util.module_from_spec(loc)
        loc.loader.exec_module(mod)
        if hasattr(mod, "evaluate"):
            _EVALUATE[name] = mod.evaluate
            spec["evaluate"] = name
    return spec, raw


def run_one(spec: dict, setting: dict, run: dict, seed: int, smoke: bool) -> dict:
    conf = _merge(setting, run.get("config"))
    if smoke:
        conf = _merge(conf, spec.get("smoke"))
    conf = _merge(conf, {"data": {"seed": seed}, "logging": {"enabled": False}})
    cfg = ts.RunConfig._from_dict(conf)
    cuts = [str(c) for c in run["cuts"]]
    cfg.discovery.n_workers = len(cuts) + 1
    m = spec["model"]

    built = []

    def model_fn():
        torch.manual_seed(seed)
        built.append(MODELS[m["name"]](**(m.get("params") or {})))
        return built[-1]

    t0 = time.perf_counter()
    hist = ts.run(model_fn, run_config=cfg, verbose=False, cuts=cuts)
    train_s = time.perf_counter() - t0
    metric = run.get("metric", spec["metric"])
    extra = {}
    if spec.get("evaluate"):
        extra = _EVALUATE[spec["evaluate"]](built[-1], cfg, run, smoke)
        hist = hist + [extra]
    values = [e[metric] for e in hist if metric in e]
    if not values and smoke and extra:       # CI without the baseline's external scorer
        return {"value": None, "scores": extra, "time_s": round(time.perf_counter() - t0, 2)}
    if not values:
        raise RuntimeError(f"run {run['label']}: no {metric!r} in the history (data.eval_every?)")
    return {"value": values[-1] if spec.get("report", "last") == "last" else max(values),
            "last": values[-1], "best": max(values), "curve": values,
            "bytes_fwd": next((e["bytes_fwd"] for e in reversed(hist) if "bytes_fwd" in e), None),
            "time_s": round(time.perf_counter() - t0, 2), "train_s": round(train_s, 2),
            **({"scores": extra} if extra else {})}


def _summary(run, rows, smoke):
    vals = [r["value"] for r in rows]
    out = {"run": run["label"], "seeds": [r["seed"] for r in rows], "per_seed": rows,
           "mean": st.mean(vals), "std": st.stdev(vals) if len(vals) > 1 else 0.0}
    exp = run.get("expected")
    if exp and not smoke:
        out["expected"] = exp
        out["within_tolerance"] = abs(out["mean"] - exp["value"]) <= exp["tol"]
    return out


def _print(results):
    print(f"{'run':24s} {'ours (mean ± std)':>20s} {'paper':>8s} {'tol':>6s}  ok")
    for r in results:
        e = r.get("expected", {})
        print(f"{r['run']:24s} {r['mean']:>12.4f} ± {r['std']:.4f} {e.get('value', float('nan')):>8.4f} "
              f"{e.get('tol', float('nan')):>6.3f}  {r.get('within_tolerance', '-')}")


def main(argv):
    name, args = argv[0], argv[1:]
    smoke = "--smoke" in args
    spec, setting = load(name)
    runs = spec["runs"]
    if "--runs" in args:
        keep = args[args.index("--runs") + 1].split(",")
        runs = [r for r in runs if r["label"] in keep]
    out_dir = os.path.join(HERE, name, "results")
    os.makedirs(out_dir, exist_ok=True)
    tag = name + ("_smoke" if smoke else "")
    final = os.path.join(out_dir, f"{tag}.json")
    if "--merge" in args or "--check" in args:
        if "--merge" in args:
            parts = sorted(f for f in os.listdir(out_dir) if f.startswith(f"{tag}.seed") and f.endswith(".json"))
            by_run = {}
            for f in parts:
                for label, row in json.load(open(os.path.join(out_dir, f)))["runs"].items():
                    by_run.setdefault(label, []).append(row)
            results = [_summary(r, sorted(by_run[r["label"]], key=lambda x: x["seed"]), smoke)
                       for r in spec["runs"] if r["label"] in by_run]
            json.dump({"baseline": spec, "setting": setting, "results": results}, open(final, "w"), indent=1)
        results = json.load(open(final))["results"]
        _print(results)
        if "--check" in args and not all(r.get("within_tolerance", True) for r in results):
            return 1
        return 0
    seeds = spec.get("seeds", [0, 1, 2])
    if smoke:
        seeds = seeds[:1]
    if "--seed" in args:
        seeds = [int(args[args.index("--seed") + 1])]
        rows = {}
        for run in runs:
            rows[run["label"]] = {"seed": seeds[0], **run_one(spec, setting, run, seeds[0], smoke)}
            print(run["label"], seeds[0], round(rows[run["label"]]["value"], 4), flush=True)
        part = "" if "--runs" not in args else "." + "+".join(sorted(rows))
        json.dump({"seed": seeds[0], "runs": rows},
                  open(os.path.join(out_dir, f"{tag}.seed{seeds[0]}{part}.json"), "w"), indent=1)
        return 0
    results = []
    for run in runs:
        rows = [{"seed": s, **run_one(spec, setting, run, s, smoke)} for s in seeds]
        results.append(_summary(run, rows, smoke))
        print(run["label"], round(results[-1]["mean"], 4), flush=True)
    json.dump({"baseline": spec, "setting": setting, "results": results}, open(final, "w"), indent=1)
    _print(results)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
