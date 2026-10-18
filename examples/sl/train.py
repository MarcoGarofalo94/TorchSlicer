"""Split-learning settings from one config: roles, cut hooks, metrics, several clients.

The model and data are those of examples/run_anywhere (a small CNN on synthetic
CIFAR-shaped data); what changes is the ``sl:`` section of the config:

    python examples/sl/train.py                                    # experiments/sl_u_shaped_p2p.yaml
    EXPERIMENT_CONFIG=experiments/sl_splitfed_local.yaml python examples/sl/train.py
    python -m torchslicer launch examples/sl/train.py -c experiments/sl_u_shaped_p2p.yaml \\
        --target processes                                         # one OS process per node

With ``sl.clients: K`` the data is split into K client shards and ``data_fn`` returns
one loader per client. For ``client_mode: parallel`` this script averages the clients'
client-side weights at the end of each round (FedAvg, as in SplitFed): averaging is a
hook (``on_round_end``), not part of TorchSlicer. See docs/tutorials/sl_settings.md.
"""

import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

import torchslicer as ts

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "run_anywhere"))
from train import build_model, get_loader   # noqa: E402  (run_anywhere's model and data)

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "..", "experiments",
                              "sl_u_shaped_p2p.yaml")


def fedavg(states, info):
    """Average the client-side weights of every client (buffers too); keep optimizers."""
    avg = {i: {k: sum(s[i]["model"][k].float() for s in states) / len(states)
               for k in states[0][i]["model"]} for i in states[0]}
    return [{i: {**s[i], "model": {k: avg[i][k].to(v.dtype) for k, v in s[i]["model"].items()}}
             for i in s} for s in states]


def get_data(cfg):
    loader = get_loader()
    k = int(cfg.sl.clients)
    if k <= 1:
        return loader
    ds = loader.dataset
    shards = [Subset(ds, range(i, len(ds), k)) for i in range(k)]      # disjoint, fixed
    loaders = [DataLoader(s, batch_size=loader.batch_size) for s in shards]
    if cfg.sl.client_mode == "parallel":
        return ts.ClientSession(loaders, mode="parallel", on_round_end=fedavg)
    return loaders                      # ts.run wraps them (sl.client_mode, sl.handoff)


if __name__ == "__main__":
    cfg = ts.RunConfig.load(os.environ.get("EXPERIMENT_CONFIG") or DEFAULT_CONFIG)
    history = ts.run(build_model, lambda: get_data(cfg), run_config=cfg, cuts=["pool1", "pool2"])
    if history:   # only the data owner (driving node) has one
        for h in history:
            print(f"epoch loss={h['loss']:.4f} bytes/batch={h['bytes_per_batch']:.0f} "
                  f"client bytes={h['bytes_client']} labels sent={h['bytes_label']} "
                  f"client compute={h['client_compute_s']:.3f}s", flush=True)
        print("HISTORY " + json.dumps([h["loss"] for h in history]), flush=True)
