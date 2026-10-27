"""P2P node for the matrix: usage run_p2p.py <model> <port>; role from IS_DRIVER.

Drives examples/train/p2p/worker/main.py with the matrix model/data (its
build_model / get_dataset / get_pack hooks).
"""
import json
import sys

import torch

import tsmodels as M
import examples.train.p2p.worker.main as p2p

name, port = sys.argv[1], sys.argv[2]
torch.use_deterministic_algorithms(True, warn_only=True)
model, pack = M.build(name)
p2p.build_model = lambda: model
p2p.get_dataset = lambda *a, **k: M.loader(name)
p2p.get_pack = lambda: pack
p2p.get_split = lambda: M.slice_kwargs(name)
sys.argv = [sys.argv[0], port]
history = p2p.serve()
if history:   # driver
    print(json.dumps({"losses": [e["loss"] for e in history]}), flush=True)
