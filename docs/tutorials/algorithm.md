# Writing an SL Algorithm

`ts.Algorithm` is the extension point for split-learning algorithms. Override
only the hooks you need; the defaults reproduce vanilla SplitNN.

The design mirrors [Flower's Strategy](https://flower.ai/docs/framework/explanation-flower-strategy-abstraction.html):
one class, one instance, called by every executor (local, centralized, P2P).

## Anatomy

```text
class Algorithm:
    # called once at setup
    def configure(self, n_slices, criterion, placement): ...

    # per-batch
    def backward_policy(self, round) -> BackwardPolicy:   # FULL or LOCAL_LOSS
    def loss(self, output, labels, ctx) -> Tensor | None  # None → use executor criterion
    def client_loss(self, cut_output, labels, ctx) -> Tensor   # LOCAL_LOSS only
    def client_cut(self) -> int | None                    # which slice is the cut
    def step_aux(self, ctx)                               # step aux optimizers

    # per-round (epoch)
    def on_round_end(self, client_states, metrics, round) -> list | None
    def aggregate(self, client_states, server_states, round) -> (dict|None, dict|None)
    def needs_per_client_server_state(self) -> bool       # SplitFed V1
```

`AlgorithmContext` carries `round`, `batch_id`, and `n_slices`.

## Built-in algorithms

| Name | Class | Registered |
|------|-------|-----------|
| `"default"` | `DefaultAlgorithm` | yes |
| `"splitfed"` | `SplitFedAlgorithm` | yes |
| `"splitfed_v1"` | `SplitFedV1Algorithm` | yes |
| — | `LocalLossAlgorithm` | no (needs `aux_head`) |

## Pass an algorithm

```python
import torch, torch.nn as nn, torchslicer as ts
torch.manual_seed(0)
_m = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
_d = [(torch.randn(4, 8), torch.randint(0, 4, (4,))) for _ in range(2)]
_c = ts.RunConfig()
_c.training.optimizer = {"name": "SGD", "params": {"lr": 0.01}}
_c.training.criterion = {"name": "CrossEntropyLoss", "params": {}}
_c.sl.setting = "vanilla"; _c.logging.enabled = False
ts.slice(_m, n=2).train(_d, run_config=_c, algorithm=ts.SplitFedAlgorithm())
```

Or name it in a config YAML:

```yaml
sl:
  algorithm: splitfed
```

## Example: label-smoothed loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchslicer as ts

class LabelSmoothAlgorithm(ts.Algorithm):
    def __init__(self, smoothing=0.1):
        self.smoothing = smoothing

    def loss(self, output, labels, ctx):
        n_classes = output.size(-1)
        smooth = self.smoothing / (n_classes - 1)
        target = torch.full_like(output, smooth).scatter_(
            1, labels.unsqueeze(1), 1.0 - self.smoothing
        )
        return F.cross_entropy(output, target.argmax(1))

torch.manual_seed(0)
_model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
_data  = [(torch.randn(4, 8), torch.randint(0, 4, (4,))) for _ in range(2)]
_cfg   = ts.RunConfig()
_cfg.training.optimizer = {"name": "SGD", "params": {"lr": 0.01}}
_cfg.training.criterion = {"name": "CrossEntropyLoss", "params": {}}
_cfg.logging.enabled = False

sliced = ts.slice(_model, n=2)
sliced.train(_data, run_config=_cfg, algorithm=LabelSmoothAlgorithm(0.1))
```

## Example: local-loss (no server gradient)

```python
import torch
import torch.nn as nn
import torchslicer as ts

torch.manual_seed(0)
_model2 = nn.Sequential(
    nn.Linear(8, 16), nn.ReLU(),    # slice 0 (client)
    nn.Linear(16, 4),               # slice 1 (server, not used)
)
_data2  = [(torch.randn(4, 8), torch.randint(0, 4, (4,))) for _ in range(2)]
_cfg2   = ts.RunConfig()
_cfg2.training.optimizer = {"name": "SGD", "params": {"lr": 0.01}}
_cfg2.training.criterion = {"name": "CrossEntropyLoss", "params": {}}
_cfg2.logging.enabled = False

aux = nn.Linear(16, 4)              # maps cut activation → logits
sliced2 = ts.slice(_model2, n=2)
sliced2.train(_data2, run_config=_cfg2,
              algorithm=ts.LocalLossAlgorithm(aux, cut_slice_idx=0))
```

`LocalLossAlgorithm` runs forward + backward only on slices 0..`cut_slice_idx`.
The server slices never execute in that batch.

## Example: FedAvg on client models (SplitFed V2)

```python
import torch
import torch.nn as nn
import torchslicer as ts

torch.manual_seed(0)
_model3  = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
_loader1 = [(torch.randn(4, 8), torch.randint(0, 4, (4,))) for _ in range(2)]
_loader2 = [(torch.randn(4, 8), torch.randint(0, 4, (4,))) for _ in range(2)]
_cfg3    = ts.RunConfig()
_cfg3.training.optimizer  = {"name": "SGD", "params": {"lr": 0.01}}
_cfg3.training.criterion  = {"name": "CrossEntropyLoss", "params": {}}
_cfg3.training.epochs     = 2
_cfg3.logging.enabled     = False
_cfg3.sl.setting          = "vanilla"

session = ts.ClientSession([_loader1, _loader2], mode="parallel")
sliced3 = ts.slice(_model3, n=2)
sliced3.train(session, run_config=_cfg3, algorithm=ts.SplitFedAlgorithm())
```

`SplitFedAlgorithm.on_round_end` averages all client-slice states with
`ts.fedavg` after every epoch and broadcasts the average back to each client.

## Registering an algorithm by name

```python
import torchslicer as ts

class LabelSmoothAlgorithm2(ts.Algorithm):
    pass

ts.register_algorithm("label_smooth2", LabelSmoothAlgorithm2)
assert "label_smooth2" in ts.available_algorithms()
assert isinstance(ts.get_algorithm("label_smooth2"), LabelSmoothAlgorithm2)
```

`get_algorithm(name)` instantiates the class with no arguments. Algorithms
that require constructor parameters (like `LocalLossAlgorithm`) should not be
registered — pass them as instances instead.

## Combine with cut hooks

Algorithms and cut hooks are orthogonal: hooks compress or protect
activations crossing the wire; the algorithm controls the training loop.

```python
import torch
import torch.nn as nn
import torchslicer as ts

torch.manual_seed(0)
_model4 = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
_data4  = [(torch.randn(4, 8), torch.randint(0, 4, (4,))) for _ in range(2)]
_cfg4   = ts.RunConfig()
_cfg4.training.optimizer = {"name": "SGD", "params": {"lr": 0.01}}
_cfg4.training.criterion = {"name": "CrossEntropyLoss", "params": {}}
_cfg4.logging.enabled    = False

sliced4 = ts.slice(_model4, n=2, cut_hooks=ts.BFloat16())
sliced4.train(_data4, run_config=_cfg4, algorithm=ts.SplitFedAlgorithm())
```
