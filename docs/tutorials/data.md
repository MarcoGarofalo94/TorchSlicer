# Data, Client Splits and Evaluation

An SL experiment is usually *a dataset, split across clients in some (non-)IID way, and a
test set to report*. With a `data` section in the config, TorchSlicer builds all three, so
`ts.run` needs no `data_fn`:

```yaml
data:
  dataset: cifar10              # mnist | fashion_mnist | cifar10 | cifar100 | synthetic | npz
  partition: dirichlet          # iid | dirichlet | shards | quantity
  partition_params: {alpha: 0.5}
  batch_size: 32
  seed: 0                       # split, subsets and shuffling
  train_subset: 0               # N samples for smoke tests (0 = all)
  eval_every: 1                 # test set every N epochs (0 = never)
sl:
  setting: vanilla
  clients: 10
  client_mode: parallel
  algorithm: splitfed
```

```python
history = ts.run(model_fn, run_config=ts.RunConfig.load("experiments/m11_cifar10_dirichlet.yaml"))
history[-1]["test_accuracy"]
```

`examples/data/train.py` is a complete script. Datasets download to `data.root` (torchvision
is used for the download only). With logging on, the split is saved as
`runs/<run_id>/client_split.json`, so a result can be traced back to exactly which samples
each client had.

## Partitioners

`ts.data.partition(labels, n_clients, method, seed, **params)` returns a `ClientSplit`: one
sorted index array per client. Same inputs and seed give the same split on any machine.

| method | params | what it does |
|---|---|---|
| `iid` | | random, near-equal sizes |
| `dirichlet` | `alpha`, `min_size` | label skew: each class's samples are shared across clients by Dirichlet(alpha); small alpha = few classes per client |
| `shards` | `shards_per_client` | pathological label skew: label-sorted shards, `shards_per_client` per client |
| `quantity` | `beta`, `min_size` | quantity skew: client sizes by Dirichlet(beta), labels IID |

```python
split = ts.data.partition(train_set, 10, "dirichlet", alpha=0.1, seed=0)
split.label_counts(train_set)          # clients x classes
loaders = split.loaders(train_set, batch_size=32)
session = ts.ClientSession(loaders, mode="parallel")
```

Add your own with `ts.data.register_partitioner(name, fn)`, where
`fn(labels, n_clients, rng, **params)` returns the index arrays.

## Evaluation

`sliced.train(..., eval_data=test_loader, eval_every=1)`, or `data.eval_every` with a
`data` section, adds `test_loss`, `test_accuracy` (classification) and `test_samples` to
the history entries.
- Forward only, no gradients, every slice in eval mode. Training is unchanged: the RNG
  state is restored afterwards, and BatchNorm statistics aren't touched.
- Cut hooks transform the activations as in training (`CutContext.direction == "eval"`),
  so a quantized or noised cut is evaluated as deployed. `Recorder` doesn't record it, and
  no bytes are counted.
- **Sessions:** each distinct client state is evaluated with the shared server slices;
  the history has the mean and `test_accuracy_clients`/`test_loss_clients`. After FedAvg
  all clients share one state, which is evaluated once.
- **Supported on the local executor** (the simulator). Distributed, P2P and offload refuse
  `eval_data` with a clear error for now. With a `data` section on those topologies, set
  `data.eval_every: 0`.
