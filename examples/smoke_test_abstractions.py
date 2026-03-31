#!/usr/bin/env python3
"""
GPU smoke-test covering everything added in the abstractions refactor:
  - ts.run() one-liner with string optimizer/criterion
  - ParameterBalancedSplitter and ExplicitSplitter
  - GPipeSchedule with micro-batches
  - UShapedTopology and PipelineTopology
  - DPNoiseHook and NoPeekHook
  - mixed_precision
  - LocalExecutor with explicit schedule
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchslicer as ts
from torchslicer.executors.local import LocalExecutor
from torchslicer.pipeline import GPipeSchedule, StandardSchedule
from torchslicer.topology import PipelineTopology, UShapedTopology
from torchslicer.hooks import DPNoiseHook, NoPeekHook

DEVICE = "cuda"

torch.manual_seed(0)
X = torch.randn(256, 64, device=DEVICE)
y = torch.randint(0, 10, (256,), device=DEVICE)
loader = DataLoader(TensorDataset(X, y), batch_size=32)


def make_model():
    return nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 10),
    ).to(DEVICE)


def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print('─'*55)


# ── 1. ts.run() one-liner with string shortcuts ───────────────────────────────
section("1. ts.run() — string optimizer + criterion")
history = ts.run(make_model(), loader, n=3, epochs=2, optimizer="adamw",
                 criterion="cross_entropy")
print(f"   epoch losses: {[round(h['loss'], 4) for h in history]}")
assert all(h["loss"] < float("inf") for h in history)
print("   OK")

# ── 2. ParameterBalancedSplitter ──────────────────────────────────────────────
section("2. ParameterBalancedSplitter")
# Three similarly-sized blocks → splitter should produce 3 partitions
pb_model = nn.Sequential(
    nn.Linear(64, 128), nn.ReLU(),   # ~8k params
    nn.Linear(128, 128), nn.ReLU(),  # ~16k params
    nn.Linear(128, 128), nn.ReLU(),  # ~16k params
    nn.Linear(128, 10),              # ~1k params
).to(DEVICE)
sliced = ts.slice(pb_model, n=3, strategy="param_balanced")
all_layers = list(pb_model.children())
sizes = [sum(p.numel() for i in part.layer_indices for p in all_layers[i].parameters())
         for part in sliced.partitions]
print(f"   param counts per partition: {sizes}  ({len(sliced.partitions)} partitions)")
assert len(sliced.partitions) == 3
print("   OK")

# ── 3. ExplicitSplitter ───────────────────────────────────────────────────────
section("3. ExplicitSplitter")
from torchslicer.strategies.explicit import ExplicitSplitter
exp_model = make_model()
graph = ts.ModelGraph.from_sequential(exp_model)
splitter = ExplicitSplitter(boundaries=[2, 4])
partitions = splitter.split(graph, n_partitions=3)
print(f"   partition layer_indices: {[p.layer_indices for p in partitions]}")
assert len(partitions) == 3
print("   OK")

# ── 4. GPipeSchedule ──────────────────────────────────────────────────────────
section("4. GPipeSchedule (n_micro=4)")
schedule = GPipeSchedule(n_micro=4)
executor = LocalExecutor(schedule=schedule)
history = ts.run(make_model(), loader, n=3, epochs=2, optimizer="adamw",
                 criterion="cross_entropy", executor=executor)
print(f"   epoch losses: {[round(h['loss'], 4) for h in history]}")
assert all(h["loss"] < float("inf") for h in history)
print("   OK")

# ── 5. mixed_precision ────────────────────────────────────────────────────────
section("5. mixed_precision (bfloat16 autocast)")
history = ts.run(make_model(), loader, n=3, epochs=2, optimizer="adamw",
                 criterion="cross_entropy", mixed_precision=True)
print(f"   epoch losses: {[round(h['loss'], 4) for h in history]}")
assert all(h["loss"] < float("inf") for h in history)
print("   OK")

# ── 6. PipelineTopology ───────────────────────────────────────────────────────
section("6. PipelineTopology")
sliced = ts.slice(make_model(), n=3)
topo = PipelineTopology()
roles = topo.assign(sliced.partitions)
print(f"   roles: { {k: len(v) for k, v in roles.items()} }")
assert len(roles["stage"]) == 3
print("   OK")

# ── 7. UShapedTopology ────────────────────────────────────────────────────────
section("7. UShapedTopology")
sliced = ts.slice(make_model(), n=3)
topo = UShapedTopology()
roles = topo.assign(sliced.partitions)
print(f"   roles: { {k: len(v) for k, v in roles.items()} }")
assert len(roles["client"]) == 2 and len(roles["server"]) == 1
print("   OK")

# ── 8. DPNoiseHook ────────────────────────────────────────────────────────────
section("8. DPNoiseHook (sigma=0.05)")
schedule = StandardSchedule(hooks=[DPNoiseHook(sigma=0.05)])
executor = LocalExecutor(schedule=schedule)
history = ts.run(make_model(), loader, n=3, epochs=2, optimizer="adamw",
                 criterion="cross_entropy", executor=executor)
print(f"   epoch losses: {[round(h['loss'], 4) for h in history]}")
assert all(h["loss"] < float("inf") for h in history)
print("   OK")

# ── 9. NoPeekHook ─────────────────────────────────────────────────────────────
section("9. NoPeekHook (lambda_=0.05)")
schedule = StandardSchedule(hooks=[NoPeekHook(lambda_=0.05)])
executor = LocalExecutor(schedule=schedule)
history = ts.run(make_model(), loader, n=3, epochs=2, optimizer="adamw",
                 criterion="cross_entropy", executor=executor)
print(f"   epoch losses: {[round(h['loss'], 4) for h in history]}")
assert all(h["loss"] < float("inf") for h in history)
print("   OK")

# ── 10. DPNoise + NoPeek combined ─────────────────────────────────────────────
section("10. DPNoiseHook + NoPeekHook combined")
schedule = StandardSchedule(hooks=[DPNoiseHook(sigma=0.02), NoPeekHook(lambda_=0.05)])
executor = LocalExecutor(schedule=schedule)
history = ts.run(make_model(), loader, n=3, epochs=2, optimizer="adamw",
                 criterion="cross_entropy", executor=executor)
print(f"   epoch losses: {[round(h['loss'], 4) for h in history]}")
assert all(h["loss"] < float("inf") for h in history)
print("   OK")

# ── 11. NoPeek + GPipe ────────────────────────────────────────────────────────
section("11. NoPeekHook + GPipeSchedule")
schedule = GPipeSchedule(n_micro=4, hooks=[NoPeekHook(lambda_=0.05)])
executor = LocalExecutor(schedule=schedule)
history = ts.run(make_model(), loader, n=3, epochs=2, optimizer="adamw",
                 criterion="cross_entropy", executor=executor)
print(f"   epoch losses: {[round(h['loss'], 4) for h in history]}")
assert all(h["loss"] < float("inf") for h in history)
print("   OK")

print(f"\n{'═'*55}")
print("  All checks passed.")
print('═'*55)
