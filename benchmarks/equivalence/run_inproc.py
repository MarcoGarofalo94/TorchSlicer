"""In-process modes: unsplit reference, LocalExecutor, LocalExecutor+GPipe, OffloadExecutor.

usage: run_inproc.py <model> <reference|local|gpipe|offload|offload-gpipe|offload-disk|offload-cpuopt>
       → prints one JSON line {"losses": [...]}
"""
import tempfile
import json
import sys

import torch
import torch.nn as nn

import torchslicer as ts
import tsmodels as M


def reference(name, cpu_opt=False):
    """``cpu_opt``: the optimizer steps on CPU copies of the parameters (the arithmetic of
    offload's CPU optimizer placement, which differs from a CUDA step by rounding)."""
    model = M.reference_module(name).to(M.DEVICE)
    if name.removesuffix("-lora") in ("gpt2", "llama"):
        # pack must reproduce the native HF forward exactly
        native, _ = M.build(name)
        native = native.to(M.DEVICE)
        ids = next(iter(M.loader(name)))[0].to(M.DEVICE)
        with torch.no_grad():
            diff = (model(ids) - native(ids).logits.transpose(1, 2)).abs().max().item()
        assert diff < 1e-5, f"pack forward differs from HF forward by {diff}"
    params = list(model.parameters())
    host = [p.detach().cpu().clone().requires_grad_(p.requires_grad) for p in params] if cpu_opt else params
    opt = getattr(torch.optim, M.OPTIMIZER["name"])(host, **M.OPTIMIZER["params"])
    crit = nn.CrossEntropyLoss()
    losses = []
    for _ in range(M.EPOCHS):
        tot = n = 0
        for x, y in M.loader(name):
            x, y = x.to(M.DEVICE), y.to(M.DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            if cpu_opt:
                for h, p in zip(host, params):
                    h.grad = None if p.grad is None else p.grad.cpu()
                    p.grad = None
            opt.step()
            if cpu_opt:
                for h, p in zip(host, params):
                    p.data.copy_(h.data)
                    h.grad = None
            tot += loss.item(); n += 1
        losses.append(tot / n)
    return losses


OFFLOAD = {"offload": {}, "offload-gpipe": {}, "offload-disk": {"mode": "disk"},
           "offload-cpuopt": {"optimizer": "cpu"}}


def sliced(name, gpipe, offload=None):
    model, pack = M.build(name)
    if offload is None:
        model, ex = model.to(M.DEVICE), None
    else:   # the model stays on the host; slices are paged into M.DEVICE
        kw = {"mode": "cpu", "optimizer": "gpu", **offload}
        if kw["mode"] == "disk":
            kw["directory"] = tempfile.mkdtemp(prefix="ts-offload-")
        ex = ts.OffloadExecutor(device=M.DEVICE, **kw)
    s = ts.slice(model, n=M.N_PARTS, pack=pack, executor=ex, **M.slice_kwargs(name))
    h = s.train(M.loader(name), M.OPTIMIZER, M.CRITERION, epochs=M.EPOCHS,
                use_gpipe=gpipe, n_micro_batches=2 if gpipe else None)
    if offload and offload.get("mode") == "disk":
        import shutil
        shutil.rmtree(kw["directory"], ignore_errors=True)
    return [e["loss"] for e in h]


if __name__ == "__main__":
    name, mode = sys.argv[1], sys.argv[2]
    torch.use_deterministic_algorithms(True, warn_only=True)
    if mode.startswith("reference"):
        losses = reference(name, cpu_opt=mode == "reference-cpuopt")
    else:
        try:
            losses = sliced(name, mode.endswith("gpipe"), OFFLOAD.get(mode))
        except ValueError as e:      # a combination refused by design
            if "shared by several slices" not in str(e):
                raise
            print(json.dumps({"losses": None, "skip": str(e)}))
            sys.exit(0)
    print(json.dumps({"losses": losses}))
