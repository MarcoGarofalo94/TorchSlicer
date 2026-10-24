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
| `natural` | `groups` (default: the dataset's) | the data's own partition (writers, speakers): whole groups per client |

**Feature shift** (same labels, different inputs per client) goes on top of any split:
`data.feature_shift: {kind: noise, level: 0.5}` (Gaussian noise, std growing from 0 on
client 0 to `level` on the last client, fixed per sample) or `{kind: rotate, level: 30}`
(degrees). In Python: `split.loaders(ds, 32, feature_shift=ts.data.FeatureShift("noise", 0.5))`.
The test set is not shifted.

**Datasets with a natural partition:** `shakespeare` is next-character prediction on
tiny-shakespeare, tagged by speaking role, so `partition: natural` gives one or more roles per
client. It follows the spirit of LEAF's Shakespeare, but the corpus and split are smaller
and different. For LEAF's FEMNIST, or any naturally partitioned data, convert it to an `.npz`
with `x`, `y` and a `groups` array (e.g. writer ids) and use `dataset: npz`.
Evaluation reports per-token accuracy for `(N, C, T)` outputs.

```python
split = ts.data.partition(train_set, 10, "dirichlet", alpha=0.1, seed=0)
split.label_counts(train_set)          # clients x classes
loaders = split.loaders(train_set, batch_size=32)
session = ts.ClientSession(loaders, mode="parallel")
```

Add your own with `ts.data.register_partitioner(name, fn)`, where
`fn(labels, n_clients, rng, **params)` returns the index arrays.

A dataset of your own works the same way: `ts.data.register_dataset(name, fn)`, with
`fn(train, root, download, seed, **params)` returning a `Dataset` of `(input, label)` pairs.
`data.dataset: name` then loads it, `data.params` are passed to `fn`, and the partitioners
split it by its `targets` (or `groups`).

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
- **Distributed runs** (centralized, P2P, client nodes) evaluate on the data owner: after
  the epoch it fetches every slice's current weights (`get_state`) into a local copy of the
  split and evaluates there, so the numbers equal the in-process run's (the tests check
  this). With client nodes, the coordinator holds the test set (a `data` section builds it
  there) and fetches each client's slice 0. The fetched bytes are reported in
  `bytes_eval`, apart from the training traffic. Not yet: per-client server copies
  (SplitFed V1) on client nodes, and offload (both refuse `eval_data`).

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

## Heterogeneous clients (simulated; deprecated)

**Deprecated, removed before 1.0** (a `DeprecationWarning` says so): simulated time depends
on hardware a simulation doesn't have. Use the hardware-independent `bytes_*`, `mem_bytes`
and `flops` of every in-process epoch, and measure time on real nodes.

Real SL clients differ in speed and network. In the simulator, give the clients profiles;
nothing is slowed down, the history just reports what the round would have cost:

```yaml
sl:
  client_profiles:
    - {name: phone,  share: 0.7, compute_scale: 8.0, bandwidth_mbps: 20,  latency_ms: 60}
    - {name: laptop, share: 0.3, compute_scale: 2.0, bandwidth_mbps: 100, latency_ms: 20}
  profile_seed: 0
```

Each client draws a profile by `share` (seeded; `client_tiers` in the first history entry).
Per round, a client's simulated time is its measured client-slice compute × `compute_scale`
plus its own bytes and messages across the data owner's boundary over its link. That's
`sim_client_time_s`, per client. `sim_round_s` is the slowest client plus the server's
compute when clients run in parallel, or the sum when they run in sequence. Distributed runs
refuse profiles: their clients are real.

The link model is `messages × latency + bytes / bandwidth`, the same as `sl.link_*`. To
simulate a real transport on your machine, measure it with `benchmarks/transport/bench_transport.py`
and fit the two numbers with `benchmarks/transport/fit_link.py`.
