"""F1 check: slicing ResNet-18.

1. Inspect: the graph with sizes (params, bytes at every possible cut), where cuts are allowed.
2. Cut: from the inspected graph (ts.slice(g)) or on the fly (ts.slice(model, ...)); same plan.
3. Look at the plan: slice summary with sizes, the slices as modules (s.slices()).
4. Exactness: every way of cutting trains like unsplit PyTorch (losses and final weights).
5. Same on a cluster: a centralized run on loopback workers ends with the same trained model
   (training.sync_back: all).

Run: .venv/bin/python notes/report/f01_slicing.py
"""
import copy
import logging

import torch
import torch.nn.functional as F
import torchvision

import torchslicer as ts

logging.getLogger("torchslicer").setLevel(logging.WARNING)   # the coordinator logs its plan at INFO
torch.manual_seed(0)
BASE = torchvision.models.resnet18(num_classes=10)
X = torch.randn(8, 3, 32, 32)
Y = torch.randint(0, 10, (8,))
STEPS, LR = 3, 0.05
OPT = {"name": "SGD", "params": {"lr": LR}}
LOSS = {"name": "CrossEntropyLoss"}            # "params" may be left out


def model():
    return copy.deepcopy(BASE).train()


def same_weights(a, b, atol=1e-5):
    return all(torch.allclose(p, q, atol=atol) for p, q in zip(a.state_dict().values(), b.state_dict().values()))


def section(title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


# ── 1. Inspect ──────────────────────────────────────────────────────────────
section("1. ts.trace(model, expand=['layer1']).summary(params=True, example_input=x)")
m = model()
g = ts.trace(m, expand=["layer1"])
print(g.summary(params=True, example_input=X))
print("valid cuts after nodes:", g.valid_cuts())

section("1b. Inside a residual block: expand=['layer1.0'] (no 'cut ok' where the skip spans)")
print(ts.trace(m, expand=["layer1.0"]).summary())

# ── 2. Cut ──────────────────────────────────────────────────────────────────
section("2. Cut the inspected graph, or on the fly: the same plan")
from_graph = ts.slice(g, cuts=["layer1.0", "layer3"])
on_the_fly = ts.slice(model(), cuts=["layer1.0", "layer3"])     # dotted cut -> expands layer1 itself
same_plan = ([p.layer_indices for p in from_graph.partitions] == [p.layer_indices for p in on_the_fly.partitions]
             and from_graph.graph.names() == on_the_fly.graph.names())
print("same partitions and nodes:", same_plan)
assert same_plan

refusals = {
    "ts.slice(g, cuts=['layer2.0'])  (g did not expand layer2)": lambda: ts.slice(g, cuts=["layer2.0"]),
    "ts.slice(g, cuts=['layer1'], expand=['layer2'])  (g is cut as traced)":
        lambda: ts.slice(g, cuts=["layer1"], expand=["layer2"]),
    "ts.slice(model, cuts=['layer1.0.bn1'])  (inside a residual block)":
        lambda: ts.slice(model(), cuts=["layer1.0.bn1"]),
}
for what, call in refusals.items():
    try:
        call()
        raise AssertionError(f"not refused: {what}")
    except ValueError as e:
        print(f"{what}\n    -> ValueError: {str(e).splitlines()[0]}")

# ── 3. Look at the plan ─────────────────────────────────────────────────────
section("3. s.summary(params=True, example_input=x) and s.slices()")
print(from_graph.summary(params=True, example_input=X))
client, *server = from_graph.slices(copy=True)        # copies: running them leaves the model alone
with torch.no_grad():
    h = client.eval().run(X)
    for s in server:
        h = s.eval().run(h)
    exact = torch.equal(h, copy.deepcopy(m).eval()(X))
print("slice 0:", client.layers)
print("slices chained == model(x):", exact)
assert exact

# ── 4. Exactness ────────────────────────────────────────────────────────────
section(f"4. Train {STEPS} SGD steps: sliced vs unsplit (losses and final weights)")
ref = model()
opt = torch.optim.SGD(ref.parameters(), lr=LR)
ref_losses = []
for _ in range(STEPS):
    opt.zero_grad()
    loss = F.cross_entropy(ref(X), Y)
    loss.backward()
    opt.step()
    ref_losses.append(loss.item())
ref_mean = sum(ref_losses) / STEPS

variants = {
    "uniform n=4": lambda mm: ts.slice(mm, n=4),
    "params n=3": lambda mm: ts.slice(mm, strategy="params", n=3),
    "flops n=3": lambda mm: ts.slice(mm, strategy="flops", n=3, example_input=X),
    "cuts on the fly [layer2, layer3.0]": lambda mm: ts.slice(mm, cuts=["layer2", "layer3.0"]),
    "cuts from ts.trace(expand=all) [add_3]": lambda mm: ts.slice(ts.trace(mm, expand="all"), cuts=["add_3"]),
}
print(f"{'cutting':<42} {'loss':>10} {'unsplit':>10} {'|diff|':>8}  {'weights':>8}  bytes fwd per cut (3 batches)")
for name, cut in variants.items():
    mm = model()
    s = cut(mm)
    hist = s.train([(X, Y)] * STEPS, OPT, LOSS, epochs=1)
    ok = same_weights(mm, ref)
    print(f"{name:<42} {hist[0]['loss']:>10.6f} {ref_mean:>10.6f} {abs(hist[0]['loss'] - ref_mean):>8.1e}  "
          f"{'equal' if ok else 'DIFFER':>8}  {hist[0]['bytes_fwd']}")
    assert ok and abs(hist[0]["loss"] - ref_mean) < 1e-5

# ── 5. Same on a cluster ────────────────────────────────────────────────────
section("5. Centralized run on 3 loopback gRPC workers: the model ends trained like unsplit")
cfg = ts.RunConfig()
cfg.training.epochs = 1
cfg.training.optimizer, cfg.training.criterion = OPT, LOSS
cfg.discovery.n_workers = 3
cfg.deploy.topology = "centralized"
cfg.network.device = "cpu"
cfg.logging.enabled = False
for sync in ("all", "none"):
    cfg.training.sync_back = sync
    mm = model()
    hist = ts.run(lambda: mm, lambda: [(X, Y)] * STEPS, run_config=cfg, verbose=False,
                  cuts=["layer2", "layer3"])
    print(f"sync_back={sync:<5} loss {hist[0]['loss']:.6f}  model trained like unsplit: {same_weights(mm, ref)}  "
          f"still initial: {same_weights(mm, BASE)}")
    assert same_weights(mm, ref) == (sync == "all")
print("\nall checks passed")
