"""Centralized coordinator for the matrix: usage run_central.py <model> <coord_port> [host].

Workers are separate processes (ts.run_worker) started by run_matrix.py; transport
comes from TENSOR_TRANSPORT (same env for coordinator and workers).
Prints one JSON line {"losses": [...]}.
"""
import json
import sys

import torch

import torchslicer as ts
from torchslicer.config import RunConfig
from torchslicer.discovery import CoordinatorDiscovery
from torchslicer.executors.distributed import DistributedExecutor
import tsmodels as M

name, port = sys.argv[1], sys.argv[2]
host = sys.argv[3] if len(sys.argv) > 3 else "127.0.0.1"
torch.use_deterministic_algorithms(True, warn_only=True)
cfg = RunConfig.load()          # env: TENSOR_TRANSPORT etc.
cfg.run_id = f"matrix-{name}"
cfg.discovery.n_workers = M.N_PARTS
cfg.discovery.timeout = 60
cfg.training.epochs = M.EPOCHS
cfg.training.optimizer = M.OPTIMIZER
cfg.training.criterion = M.CRITERION
cfg.logging.enabled = False

addr = f"{host}:{port}"
ex = DistributedExecutor(discovery=CoordinatorDiscovery(run_id=cfg.run_id),
                         coordinator_addr=addr, coordinator_bind_addr=addr, run_config=cfg)
model, pack = M.build(name)
h = ts.slice(model, n=M.N_PARTS, executor=ex, pack=pack).train(M.loader(name), run_config=cfg)
print(json.dumps({"losses": [e["loss"] for e in h]}), flush=True)
