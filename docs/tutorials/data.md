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

## Many clients

A session keeps one state per client: its client slices' weights and optimizer state,
about `(1 + optimizer slots) × client-slice parameters × 4` bytes. Switching clients
copies one state out of the slices and the next one in.

- **`sl.client_states`**: where the states live. `cpu` (host RAM, the default), `gpu`
  (`cuda:i`), or `auto` (the slices' GPU if they fit, else RAM). With GPU slices, the GPU
  store makes a switch a device-to-device copy: 0.35 ms instead of 11 ms for a 2.1M-parameter
  client slice on an A100 (`benchmarks/clients/`).
- **TorchSlicer checks before training** that N states fit in 80 % of the free memory of
  the chosen store (for the host: the smaller of the cgroup limit's remainder and
  `MemAvailable`), and stops with the numbers otherwise. A machine that runs out of memory
  mid-run, or a container that gets killed, loses more than the run.
- **Client sampling:** override `Algorithm.select_clients(available, round)` to return the
  round's participants; only they train. `on_round_end` gets their states, and
  `aggregate`'s client state (e.g. FedAvg of the participants) goes to every client, as
  in FedAvg. The history entry lists the round's `clients` when not all took part.

```python
class TenPerRound(ts.SplitFedAlgorithm):
    def select_clients(self, available, round):
        rng = random.Random(round)
        return sorted(rng.sample(available, 10))
```

Measured on one A100 with a 16 GB container (SplitFed V2, 2.1M-parameter client slices,
one batch of 16 per client):

| clients | store | round | per client | peak RAM | peak GPU |
|---|---|---|---|---|---|
| 200 | cpu | 0.74 s | 3.7 ms | 4.8 GB | 0.04 GB |
| 200 | gpu | 0.29 s | 1.4 ms | 0.9 GB | 4.0 GB |
| 1000 | gpu | 1.49 s | 1.5 ms | 1.0 GB | 19.6 GB |
| 1000 | cpu | refused before training: 15.6 GB needed, 14.8 GB free | | | |

Clients with a small client slice (a conv layer or two) are cheap anywhere: 1000 clients of
the example CNN take 2 s per round and 1.2 GB.
