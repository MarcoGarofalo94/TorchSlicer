"""In-process modes: unsplit reference, LocalExecutor, LocalExecutor+GPipe.

usage: run_inproc.py <model> <reference|local|gpipe>   → prints one JSON line {"losses": [...]}
"""
import json
import sys

import torch
import torch.nn as nn

import torchslicer as ts
import tsmodels as M


def reference(name):
    model = M.reference_module(name).to(M.DEVICE)
    if name.removesuffix("-lora") in ("gpt2", "llama"):
        # pack must reproduce the native HF forward exactly
        native, _ = M.build(name)
        native = native.to(M.DEVICE)
        ids = next(iter(M.loader(name)))[0].to(M.DEVICE)
        with torch.no_grad():
            diff = (model(ids) - native(ids).logits.transpose(1, 2)).abs().max().item()
        assert diff < 1e-5, f"pack forward differs from HF forward by {diff}"
    opt = getattr(torch.optim, M.OPTIMIZER["name"])(model.parameters(), **M.OPTIMIZER["params"])
    crit = nn.CrossEntropyLoss()
    losses = []
    for _ in range(M.EPOCHS):
        tot = n = 0
        for x, y in M.loader(name):
            x, y = x.to(M.DEVICE), y.to(M.DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item(); n += 1
        losses.append(tot / n)
    return losses


def sliced(name, gpipe):
    model, pack = M.build(name)
    model = model.to(M.DEVICE)
    s = ts.slice(model, n=M.N_PARTS, pack=pack)
    h = s.train(M.loader(name), M.OPTIMIZER, M.CRITERION, epochs=M.EPOCHS,
                use_gpipe=gpipe, n_micro_batches=2 if gpipe else None)
    return [e["loss"] for e in h]


if __name__ == "__main__":
    name, mode = sys.argv[1], sys.argv[2]
    torch.use_deterministic_algorithms(True, warn_only=True)
    losses = reference(name) if mode == "reference" else sliced(name, mode == "gpipe")
    print(json.dumps({"losses": losses}))
