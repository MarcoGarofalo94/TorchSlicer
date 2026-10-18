# Privacy: Attacks, Defences and Leakage Metrics

A split-learning paper's privacy section usually has four parts: an attack, a defence, a
leakage number and the accuracy cost. In TorchSlicer each part is one line of config, and
the numbers land in every epoch's history next to `loss` and `test_accuracy`.

- An **attack** observes one cut from the side that *receives* it. A `forward` attack sees
  the activation (what the server gets from the client). A `backward` attack sees the
  gradient (what the client gets back). It sees what that side really receives, after the
  cut hooks, so `GaussianNoise`, `Quantize` or `TopK` on that cut defend against it.
- The executor also hands the attack the batch's true inputs and labels, **for scoring
  only**. An attack must not use them to attack, and the built-in ones don't.
- A **defence** either changes what crosses the cut, and is then a cut hook (`sl.cut_hooks`),
  or adds a term to the client's training loss: `NoPeek`.
- Attacks never change training. Their randomness runs in a forked RNG, and their queries
  leave the slices exactly as they were, so the losses are bit-identical with or without
  them.

Privacy runs in the simulator: the local executor, and offload except for query-based
attacks. The distributed executors refuse it, because a real remote observer has no
ground truth to score against.

## Built in

| name (config) | class | cut side | metrics (`leak_<name>_*`) |
|---|---|---|---|
| `label_norm` | `NormLabelAttack(cut, positive=1)` | gradient | `auc`, `auc_q95`, `batches` |
| `label_direction` | `DirectionLabelAttack(cut, positive=1)` | gradient | `auc`, `auc_q95`, `batches` |
| `inversion` | `InversionAttack(cut, aux=, n_aux=512, steps=2, ...)` | activation | `psnr`, `ssim` |
| `dcor` | `DistanceCorrelation(cut)` | activation | `value` |
| `nopeek` (defence) | `NoPeek(cut, alpha)` | client loss | none (add `dcor` to measure it) |

- **Label inference** follows Li et al., ICLR 2022 (https://arxiv.org/abs/2102.08504). The
  client scores each sample by its cut gradient's norm, or by its cosine with one
  positive's gradient (the paper's oracle evaluation). The leak AUC is computed per batch;
  the epoch reports its mean and, as in the paper, its 95 % quantile.
- **Inversion** is a black-box, learning-based attack in the spirit of He et al.,
  ACSAC 2019 (https://doi.org/10.1145/3359789.3359824). The server holds auxiliary inputs
  from the data distribution (default: the run's evaluation set). It queries the client
  slices with them, trains a decoder from activations back to inputs while training runs,
  and decodes every activation it receives. PSNR and SSIM use the true batch's max - min
  as the data range.
- **NoPeek** (https://arxiv.org/abs/2008.09161) trains the client slices on
  `task loss + alpha * dcor(x, z)`, the distance correlation between the raw inputs and
  the cut activation. The client adds that term's gradient to the one the server returns.

## Label leakage, and gradient noise against it

An imbalanced binary task (10 % positives), two slices: the client learns the labels from
the gradients it gets back.

```python
import torch, torch.nn as nn
import torchslicer as ts

g = torch.Generator().manual_seed(0)
x = torch.randn(1024, 20, generator=g)
score = x @ torch.randn(20, generator=g)
y = (score > score.quantile(0.9)).long()
data = [(x[i:i + 64], y[i:i + 64]) for i in range(0, 1024, 64)]

def train(hooks=None, privacy=()):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
    sliced = ts.slice(model, cuts=["1"], cut_hooks=hooks)
    return sliced.train(data, {"name": "SGD", "params": {"lr": 0.1}},
                        {"name": "CrossEntropyLoss", "params": {}}, epochs=2,
                        privacy=[ts.privacy.NormLabelAttack(), ts.privacy.DirectionLabelAttack(), *privacy])

plain = train()[-1]
noisy = train(hooks={0: ts.GaussianNoise(std=0.05, grad=True)})[-1]
print(plain["leak_label_norm_auc"], noisy["leak_label_norm_auc"])        # ~1.0, then ~0.5
assert plain["leak_label_norm_auc"] > 0.95 > 0.7 > noisy["leak_label_norm_auc"]
```

## Distance correlation and NoPeek

```python
plain = train(privacy=[ts.privacy.DistanceCorrelation()])[-1]
nopeek = train(privacy=[ts.privacy.DistanceCorrelation(), ts.privacy.NoPeek(alpha=1.0)])[-1]
print(plain["leak_dcor_value"], nopeek["leak_dcor_value"], plain["loss"], nopeek["loss"])
assert nopeek["leak_dcor_value"] < plain["leak_dcor_value"]
```

The reported `loss` is the task loss in both runs, so the two losses show the defence's
accuracy cost. Distance correlation depends on the batch size, so compare runs with equal
batch sizes. Under GPipe it is computed per micro-batch.

## From a config

```yaml
privacy:
  attacks:
    - {name: inversion, cut: 0, params: {n_aux: 512, steps: 2}}   # aux = the test set
    - {name: dcor, cut: 0}
  defences:
    - {name: nopeek, cut: 0, params: {alpha: 0.5}}
sl:
  cut_hooks:
    - {cut: 0, name: gaussian_noise, params: {std: 0.1}}          # activation noise
```

Two attacks with the same name on different cuts report as `leak_<name>_cut<k>_*`, and
`key=` sets the name yourself.

## Your own attack

Subclass `ts.privacy.Attack`, set `name` and `direction`, and implement `observe` and
`metrics`. `ts.privacy.register_attack(name, cls)` makes it available to config files. An
attack that needs to query the client sets `needs_query = True` and gets
`self.query(x)`: black-box access to what the receiver would get for inputs `x`.

```python
class MeanActivation(ts.privacy.Attack):
    name, direction = "mean_act", "forward"

    def reset(self):
        self.values = []

    def observe(self, obs):                 # obs.tensor, obs.ctx, obs.inputs, obs.labels
        self.values.append(obs.tensor.abs().mean().item())

    def metrics(self):
        return {"value": sum(self.values) / len(self.values)} if self.values else {}

h = train(privacy=[MeanActivation()])
assert "leak_mean_act_value" in h[-1]
```

A defence subclasses `ts.privacy.Defence` and returns, from `grad_term(z, obs)`, the extra
gradient of the client's loss with respect to its cut activation `z`.
