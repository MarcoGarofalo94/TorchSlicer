# Choosing where to cut, and writing your own cutting strategy

`ts.slice(model, ...)` does three things:

1. **Trace** the model into a graph of nodes (`ts.trace(model)` does just this step).
2. **Split**: a *strategy* (a splitter) groups the nodes into consecutive partitions.
3. **Validate**: the partitions are checked, so that each one gets exactly one tensor from the
   one before it.

This page covers how to inspect the graph, the built-in ways to choose cuts, and how to
write and register your own strategy.

## 1. Look at the graph

```python
import torch, torchvision
import torchslicer as ts

model = torchvision.models.resnet18(num_classes=10)
print(ts.trace(model).summary())
```

```text
ModelGraph(source=traced, nodes=11, expand=[])
    0  conv1                        Conv2d               <- []  | cut ok
    1  bn1                          BatchNorm2d          <- [0]  | cut ok
    ...
    7  layer4                       Sequential           <- [6]  | cut ok
    8  avgpool                      AdaptiveAvgPool2d    <- [7]  | cut ok
    9  flatten                      _FlattenWrapper      <- [8]  | cut ok
   10  fc                           Linear               <- [9]
```

Each node is an `nn.Module`: a child module of the model, or a wrapped functional op
(`flatten`, `add`, `F.relu`, `x.view(...)`, ...). `<- [...]` lists the nodes it reads from.
`cut ok` means a cut right after this node is allowed: that node's output is the only value
the rest of the model needs, so one tensor carries everything across the cut. A cut inside a
skip connection is never allowed (`graph.valid_cuts()` returns the allowed positions).

**Granularity.** By default the model's direct children are the nodes, and they are not traced
into. So `layer1`..`layer4` are atomic. Use `expand=` to trace into submodules:

```python
g = ts.trace(model, expand=["layer3"])   # layer3 becomes layer3.0, layer3.1
g = ts.trace(model, expand="all")        # down to the built-in torch.nn layers
```

Inside a fully expanded residual block only the block boundaries are valid cuts. The nodes
inside are listed, but not marked `cut ok`.

## 2. Built-in ways to cut

| How | What it balances | Example |
|---|---|---|
| `strategy="uniform"` (default) | node count | `ts.slice(model, n=4)` |
| `strategy="params"` | parameter count (weights + optimizer memory) | `ts.slice(model, strategy="params", n=4)` |
| `strategy="flops"` | forward FLOPs on an example batch | `ts.slice(model, strategy="flops", n=4, example_input=x)` |
| `strategy="profiled"` | measured forward+backward time per node | `ts.slice(model, strategy="profiled", n=4, example_input=x)` |
| `cuts=[...]` | nothing: you say where | `ts.slice(model, cuts=["layer2", "layer3.1"])` |

- The cost strategies (`params`, `flops`, `profiled`) choose among the valid cuts and minimize
  the cost of the most expensive partition. That partition is the pipeline bottleneck. They
  combine well with `expand=`, which gives them more places to cut.
- `flops` and `profiled` run the model on a **copy**, so the real model's weights, BatchNorm
  statistics and RNG state are left untouched. A split chosen this way trains exactly like the
  unsplit model. Run `profiled` on the hardware the slices will use (`ts.ProfiledSplitter(x,
  device="cuda")`).
- **Named cuts** cut right after each listed node. A dotted name traces into its parent
  automatically. `cuts=["layer3.0"]` expands `layer3` and cuts between its two blocks. The
  name of a module that has been expanded means "after its output". If a cut is not allowed,
  the error names the nearest valid cuts before and after it.

```python
x = torch.randn(8, 3, 32, 32)
sliced = ts.slice(model, strategy="flops", n=3, example_input=x,
                  expand=["layer1", "layer2", "layer3", "layer4"])
print(sliced.summary())

sliced = ts.slice(model, cuts=["layer2", "layer3.0"])   # 3 partitions
print(sliced.summary())
```

## 3. Write your own strategy

A strategy subclasses `ts.BaseSplitter` and implements `split(model_graph, n_partitions)`, which
returns a list of `ts.Partition`. The easiest approach is to choose cut positions among
`model_graph.valid_cuts()` and let `partitions_from_cuts` build the partitions:

```python
class SmallestActivationCut(ts.BaseSplitter):
    """Two-party split learning: cut where the activation sent over the network is smallest,
    keeping at least `min_client_layers` nodes on the client."""

    def __init__(self, example_input=None, min_client_layers=1):
        self.example_input = example_input      # ts.slice(..., example_input=x) fills this in
        self.min_client_layers = min_client_layers

    def split(self, model_graph, n_partitions):
        assert n_partitions == 2, "this strategy makes a client/server split"
        with torch.no_grad():
            outs = model_graph.run_nodes(self.example_input)   # every node's output
        candidates = [k for k in model_graph.valid_cuts() if k + 1 >= self.min_client_layers]
        best = min(candidates, key=lambda k: outs[k].numel() * outs[k].element_size())
        return self.partitions_from_cuts(model_graph, [best])

ts.register_strategy("smallest-activation", SmallestActivationCut)

model.eval()   # run_nodes runs the real modules: eval() keeps BatchNorm statistics unchanged
sliced = ts.slice(model, strategy="smallest-activation", n=2,
                  example_input=torch.randn(2, 3, 32, 32), expand="all")
model.train()
print(sliced.summary())
```

To *balance* a per-node cost, subclass `ts.CostSplitter` and implement only `node_costs`.
The optimal min-max cut search is inherited:

```python
class ActivationMemorySplitter(ts.CostSplitter):
    """Balance the activation memory each partition keeps for backward."""

    def __init__(self, example_input=None):
        self.example_input = example_input

    def node_costs(self, model_graph):
        with torch.no_grad():
            outs = model_graph.run_nodes(self.example_input)
        return [o.numel() * o.element_size() if torch.is_tensor(o) else 0 for o in outs]

model.eval()
sliced = ts.slice(model, strategy=ActivationMemorySplitter(torch.randn(2, 3, 32, 32)), n=3)
model.train()
```

What `ts.slice` does for you, and what a strategy must respect:

- **Validation runs after `split`.** Partitions must be non-empty, contiguous and in order.
  Every edge between partitions must come from the last node of the previous partition, and
  only partition 0 may read the model input. A partition that breaks a rule raises a
  `ValueError` that lists the valid cuts.
- **Tied weights** (one parameter used in several partitions) are handled by the executors.
  A strategy only deals in node indices.
- **Finer graphs:** override `expand_for(self, model)` to return the submodules your strategy
  needs traced into (`NamedCutSplitter` does this for dotted names). Or let the user pass
  `expand=`.
- **Inputs:** if your strategy has an `example_input` attribute, `ts.slice(...,
  example_input=x)` sets it. `model_graph.run_nodes(x)` returns every node's output.
  `run_nodes` runs the real modules, so it changes BatchNorm running statistics in train mode
  and draws from the RNG for dropout. Either use `eval()` and `no_grad()` as above, or work on
  a copy, as the built-in `flops` and `profiled` strategies do.
- `strategy=` takes a registered name, a class (instantiated with no arguments), or an instance.
  `ts.available_strategies()` lists the registered names.

## 4. Check that your split trains like the unsplit model

Splitting must never change the math. With a fixed seed, the sliced model's losses must match
plain PyTorch:

```python
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

def make():
    torch.manual_seed(0)
    return torchvision.models.resnet18(num_classes=10)

g = torch.Generator().manual_seed(1)
data = TensorDataset(torch.randn(32, 3, 32, 32, generator=g), torch.randint(0, 10, (32,), generator=g))
loader = DataLoader(data, batch_size=8)
opt_cfg = {"name": "SGD", "params": {"lr": 0.05}}
crit_cfg = {"name": "CrossEntropyLoss", "params": {}}

ref = make()
opt = torch.optim.SGD(ref.parameters(), lr=0.05)
ref_losses = []
for xb, yb in loader:
    opt.zero_grad()
    loss = nn.functional.cross_entropy(ref(xb), yb)
    loss.backward()
    opt.step()
    ref_losses.append(loss.item())

sliced = ts.slice(make(), strategy="params", n=3, expand="all")
hist = sliced.train(loader, opt_cfg, crit_cfg, epochs=1)
assert abs(hist[0]["loss"] - sum(ref_losses) / len(ref_losses)) < 1e-5
```

For the full model × execution-mode check (local, GPipe, centralized and P2P, over gRPC or
TCP), see `benchmarks/equivalence/`.

## 5. Plan for heterogeneous workers

Distributed executors plan again once they know the workers. If your `split` accepts a
`nodes=None` argument, it receives the discovered workers as `ts.NodeInfo` objects, in slice
order. Each one carries `compute_gflops`, `memory_cap_mb()`, `device`, and the measured
`rtt_ms`/`bandwidth_mbps` (plus `link_*` to the next worker with
`planning.measure_links`). The built-in cost strategies already use these fields, through
`ts.balanced_cuts(costs, valid_cuts, n, speeds=..., mem=..., caps=..., ...)`. Here is a
strategy that balances node count weighted by speed:

```python
class SpeedWeighted(ts.BaseSplitter):
    def split(self, model_graph, n_partitions, nodes=None):
        costs = [1.0] * len(model_graph)
        speeds = [nd.compute_gflops for nd in nodes] if nodes else None
        cuts = ts.balanced_cuts(costs, model_graph.valid_cuts(), n_partitions, speeds=speeds)
        return self.partitions_from_cuts(model_graph, cuts)

graph = ts.trace(nn.Sequential(*[nn.Linear(8, 8) for _ in range(8)]))
nodes = [ts.NodeInfo("a", "a:1", compute_gflops=10), ts.NodeInfo("b", "b:1", compute_gflops=30)]
print([p.layer_indices for p in SpeedWeighted().split(graph, 2, nodes=nodes)])
# [[0, 1], [2, 3, 4, 5, 6, 7]]: the 3x faster worker gets 3x the layers
```

Without `nodes` (local runs, or `planning.heterogeneous: false`), the same strategy splits
evenly.
