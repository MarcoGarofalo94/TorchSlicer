# TorchSlicer

<img src="docs/torchslicer-logo.png" width="156" height="156" />

**A split-learning framework for PyTorch.** TorchSlicer cuts any `nn.Module` into a chain of
slices and trains them end to end. The slices can run in one process (a simulator), on
gRPC workers driven by a coordinator, or peer to peer, and no node has to hold the whole
model. On top of that chain it adds what split-learning (SL) research needs: data-owner
roles (vanilla, U-shaped), hooks on every cut (compression, noise, attack observers),
multi-client sessions with SplitFed-style algorithms, federated data splits, and
communication and compute metrics in every epoch.

```python
import torchslicer as ts

sliced = ts.slice(model, n=4)                        # trace with torch.fx, cut into 4 slices
history = sliced.train(loader,
                       {"name": "SGD", "params": {"lr": 0.01}},
                       {"name": "CrossEntropyLoss", "params": {}},
                       epochs=10)
history[-1]["loss"], history[-1]["bytes_fwd"]        # loss, bytes per cut
```

Splitting doesn't change the math. On CPU every model in the validation matrix (DNN, CNN,
ResNet-18, TinyGPT, GPT-2, Llama, plus LoRA on the language models) gives the same losses
as unsplit PyTorch with the same seed, in every executor, topology and transport
([`benchmarks/equivalence/`](benchmarks/equivalence/)).
The one known difference is GPipe with BatchNorm, which micro-batching causes by design.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Features at a glance](#features-at-a-glance)
- [Split anything](#split-anything)
- [Run anywhere](#run-anywhere)
- [Split-learning settings](#split-learning-settings)
- [Algorithms and multiple clients](#algorithms-and-multiple-clients)
- [Data, client splits and evaluation](#data-client-splits-and-evaluation)
- [Transport, NAT and fault tolerance](#transport-nat-and-fault-tolerance)
- [Out-of-core training](#out-of-core-training)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Docker](#docker)
- [Repository layout](#repository-layout)
- [Development](#development)
- [Citing](#citing) · [License](#license)

---

## Installation

```bash
pip install torchslicer                            # core: local executor, simulator
pip install 'torchslicer[grpc]'                    # + distributed (centralized, P2P)
pip install 'torchslicer[grpc,monitor,peft]'       # + OpenTelemetry tracing + LoRA
```

Python ≥ 3.10. PyTorch is deliberately not a dependency: install the build that matches
your hardware first. The datasets in `ts.data` use `torchvision` for downloading.

From source (the development version, 0.11.0):

```bash
git clone https://github.com/MarcoGarofalo94/TorchSlicer && cd TorchSlicer
pip install -e '.[grpc]'
```

## Quick start

```python
import torch, torchvision
import torchslicer as ts

model = torchvision.models.resnet18(num_classes=10)
print(ts.trace(model).summary())          # the traced graph and where a cut is allowed

sliced = ts.slice(model, strategy="params", n=2)
print(sliced.summary())                   # nodes and parameters per slice

history = sliced.train(train_loader,
                       {"name": "SGD", "params": {"lr": 0.01, "momentum": 0.9}},
                       {"name": "CrossEntropyLoss", "params": {}},
                       epochs=5, eval_data=test_loader, verbose=True)
```

Each entry of `history` is one epoch: `loss`, `test_accuracy`, bytes per cut and
direction, client and server compute time, simulated link time, and more.

The same model distributed, with a YAML config and one call that works on every node:

```python
# train.py
import torchslicer as ts

def build_model(): ...       # called only where training is driven
def get_loader(): ...        # workers never see the data

if __name__ == "__main__":
    ts.run(build_model, get_loader, strategy="params")
```

```bash
python train.py                                                          # every node in this process
python -m torchslicer launch train.py -c run.yaml --target processes      # one OS process per node
python -m torchslicer launch train.py -c run.yaml --target k8s            # or compose, baremetal
```

## Features at a glance

| Area | What you get | Guide |
|---|---|---|
| **Split anything** | `torch.fx` tracing, `torch.export` for models fx can't trace (Hugging Face transformers, no `pack=`), both mixed module by module, named cuts, `expand=`, strategies `uniform`/`params`/`flops`/`profiled`/`memory`, custom strategies, `pack=` for untraceable models (HF LLMs, MoE), tied weights, cuts through skip connections (`multi_tensor=True`: U-Nets, encoder-decoders; masks rebuilt, not sent) | [custom_strategy](docs/tutorials/custom_strategy.md), [pack](docs/tutorials/pack.md), [multi_tensor](docs/tutorials/multi_tensor.md), [export](docs/tutorials/export.md) |
| **Run anywhere** | local / centralized / P2P topologies; one script + one config on local, processes, Docker Compose, Kubernetes, bare metal | [docs/deploy](docs/deploy/README.md), [p2p](docs/tutorials/p2p.md) |
| **Capability-aware planning** | workers report device, memory and measured GFLOPS; links measured; cuts balanced for heterogeneous speeds, memory caps and link cost | [p2p](docs/tutorials/p2p.md) |
| **SL settings** | `pipeline` / `vanilla` / `u_shaped` roles, `CutHook`s (fp16/bf16, quantize, top-k, Gaussian noise, recorder), SL metrics | [sl_settings](docs/tutorials/sl_settings.md) |
| **Algorithms** | `ts.Algorithm` (Flower-Strategy-style): vanilla SplitNN, SplitFed V1/V2, local loss, FedAvg, client sampling | [algorithm](docs/tutorials/algorithm.md) |
| **Multiple clients** | `ts.ClientSession`: sequential or parallel clients against shared server slices | [sl_settings](docs/tutorials/sl_settings.md) |
| **Data** | MNIST, Fashion-MNIST, CIFAR-10/100, synthetic, `.npz`; iid / Dirichlet / shards / quantity splits; test-set evaluation | [data](docs/tutorials/data.md) |
| **Flower interop** | any Flower strategy (FedAvg, FedMedian, Krum, FedAdam, ...) as the SplitFed aggregation of the client (and server) slices; Flower FedAvg = `ts.SplitFedAlgorithm` exactly; TorchSlicer clients run by Flower's own engine against a `ts.SplitServer` | [flower](docs/tutorials/flower.md) |
| **Baselines** | published SL results rerun from one command with the paper's number and a tolerance (SLPerf: vanilla, U-shaped, SplitFed) | [baselines](docs/baselines.md) |
| **Privacy** | label inference from gradients (norm / direction), learning-based input inversion, a malicious server (FSHA), architecture-only inversion (UnSplit), LLM hidden-state inversion, distance correlation; NoPeek, noise and clipped-noise defences with a DP accountant; `leak_*` metrics per epoch | [privacy](docs/tutorials/privacy.md) |
| **Transport** | gRPC, raw TCP (pinned memory), CUDA IPC, NCCL | [transport](docs/tutorials/transport.md) |
| **NAT traversal** | clients that only dial out, tunnels, optional relay | [nat](docs/tutorials/nat.md) |
| **Fault tolerance** | heartbeats, worker error reports, re-planning for the survivors from the last checkpoint | below |
| **Out-of-core** | `ts.OffloadExecutor`: train a model larger than the GPU on one node (CPU or disk tier) | [offload](docs/tutorials/offload.md) |
| **Mixed precision / pipelining** | bf16 autocast, GPipe micro-batches | below |
| **Observability** | JSONL run logs, per-phase profiling, OpenTelemetry/Phoenix traces | below |

Every code block in `docs/tutorials/` is run by the test suite (`tests/test_docs.py`).

---

## Split anything

`ts.slice(model, ...)` traces the model with `torch.fx` into a graph of named nodes, groups
the nodes into consecutive partitions with a *strategy*, and validates that exactly **one
tensor** crosses each cut.

```python
ts.slice(model, n=4)                                                   # uniform (default)
ts.slice(model, cuts=["layer2", "layer3.0"])                           # named cuts; dotted names trace inside
ts.slice(model, strategy="params", n=4)                                # balance parameters
ts.slice(model, strategy="flops", n=4, example_input=x, expand="all")  # balance FLOPs, finer graph
ts.slice(model, strategy="profiled", n=4, example_input=x)             # balance measured time
```

- Direct children are leaves unless listed in `expand=` (or `expand="all"`). Functional ops
  (`torch.flatten`, `+`, `reshape`, …) become nodes too, so nothing is dropped.
- Cost strategies solve the min-max split exactly over the valid cuts. Those that measure
  work on a copy, so model state and RNG stay untouched.
- Your own strategy: subclass `ts.BaseSplitter` and `ts.register_strategy("name", cls)`
  ([tutorial](docs/tutorials/custom_strategy.md)).

**When tracing fails** (data-dependent control flow, several inputs or outputs, most Hugging
Face models), `ts.slice` raises `ts.TraceError` instead of guessing. Pass `pack=`, a
function that returns the model as a list of one-tensor-in, one-tensor-out stages
([tutorial](docs/tutorials/pack.md); ready-made packs for GPT-2, Llama, BERT and DeepSeek-MoE
in [`examples/pack/`](examples/pack/)):

```python
def pack_qwen(model):
    return [
        ts.SimpleEmbedStage(model.model.embed_tokens),
        *[ts.BlockStage(layer) for layer in model.model.layers],
        ts.CausalLMHeadStage(model.model.norm, model.lm_head),
    ]

sliced = ts.slice(model, n=4, pack=pack_qwen)
```

**Tied weights** (e.g. the token embedding and LM head of GPT-2, Llama-3.2, Qwen2.5, Gemma)
train exactly as in the unsplit model. The lowest slice that uses the parameter owns it;
across workers the other holders send it their gradient, and it sends back the new value
each batch.

**LoRA:** `ts.slice(ts.peft_unwrap(get_peft_model(model, lora_cfg)), n=2)`.

## Run anywhere

An experiment is **one script and one YAML config**. `ts.run` reads the config
(`EXPERIMENT_CONFIG` plus env overrides) and does what this node's role says:

| `deploy.topology` | roles (`deploy.role`, env `ROLE`) | |
|---|---|---|
| `local` | `all` | `LocalExecutor`, one process, no network |
| `centralized` | `coordinator`, `worker`, `all` | a coordinator ships the slices and drives; workers chain activations peer to peer |
| `p2p` | `driver`, `follower`, `all` | the driver holds the data and slice 0 and drives; no coordinator process |

`ROLE=all` (the default) runs every node in one process, with the workers on loopback over
real gRPC/TCP. That's the easiest way to debug a distributed run.

`python -m torchslicer launch SCRIPT -c CONFIG --target T` builds the node list once from the
config and then runs it or renders files for it, under `runs/deploy/`:

| target | does |
|---|---|
| `local` | runs `ROLE=all` here |
| `processes` | starts one OS process per node on this machine |
| `compose` | renders a Docker Compose file |
| `k8s` | renders a ConfigMap, an Indexed Job of workers behind a headless Service, and a coordinator/driver Job |
| `baremetal` | renders one start script per host |

On CPU, every target gives the losses of unsplit PyTorch (`tests/test_deploy.py`).
Guides: [local](docs/deploy/local.md), [compose](docs/deploy/compose.md),
[k8s](docs/deploy/k8s.md), [baremetal](docs/deploy/baremetal.md). Example:
[`examples/run_anywhere/`](examples/run_anywhere/) with
`experiments/run_anywhere_{local,centralized,p2p}.yaml`.

Lower-level entry points: `ts.run_p2p(model_fn, data_fn, ...)`, `ts.run_worker(...)` /
`ts.start_worker()`, and the executors themselves (`ts.LocalExecutor`,
`ts.DistributedExecutor`, `ts.P2PExecutor`).

**Planning** happens after discovery. Workers report their device, free memory and
measured (or declared) GFLOPS, and can cap their memory or ask for slice 0
(`network.worker_compute_gflops`, `worker_max_memory_mb`, `worker_first_slice`). The executor
measures RTT and bandwidth, and strategies that accept `nodes=` balance the cuts for
heterogeneous speeds, memory caps and per-cut link time.

## Split-learning settings

The *data owner* is the process that holds the data: the local process, the centralized
coordinator, or the P2P driver. `sl.setting` decides which slices it hosts in-process,
and so which tensors leave it:

| `sl.setting` | data owner hosts | raw inputs | labels | usual name |
|---|---|---|---|---|
| `pipeline` | nothing | leave | leave | pipeline parallelism (centralized default) |
| `vanilla` | slice 0 | stay | leave | SL with label sharing (default elsewhere) |
| `u_shaped` | slice 0 and the last slice | stay | stay | U-shaped SL, no label sharing |

A setting never changes the math, only where each slice runs.

**Cut hooks** transform what crosses cut `k` (slice `k` → `k+1`) in both directions. The
payload is packed into one tensor, so the byte counts are what really goes on the wire:

```python
ts.run(build_model, get_data, cuts=["1", "3"],
       cut_hooks={0: ts.Quantize(bits=4)})                    # only the client → server cut

rec = ts.Recorder(what="backward", max_batches=4)            # what an attacker at the cut sees
ts.run(build_model, get_data, cuts=["1", "3"], cut_hooks={0: rec})
```

Built-ins: `Float16`, `BFloat16`, `Quantize` (1–8 bits), `TopK`, `CircularConv` (C3-SL),
`GaussianNoise`, `Recorder`, `Chain`. Write your own by subclassing `ts.CutHook` (`encode`/`decode`,
`encode_grad`/`decode_grad`), or name built-ins in the config:

```yaml
sl:
  setting: u_shaped
  cut_hooks:
    - {cut: 0, name: quantize, params: {bits: 8}}
    - {cut: all, name: gaussian_noise, params: {std: 0.01}}
```

**Metrics in every history entry**, identical locally and distributed: `bytes_fwd` /
`bytes_bwd` per cut, `bytes_input`, `bytes_label`, `bytes_client` (everything crossing the
data owner's boundary), `raw_bytes_total` (before encoding), client and server
`compute_s`, and simulated link time from `sl.link_*`.

## Algorithms and multiple clients

`ts.Algorithm` is the extension point for SL algorithms, modelled on Flower's Strategy.
Override only the hooks you need: loss, backward policy (full or local loss), per-round
aggregation, client sampling. The defaults are vanilla SplitNN, bit-exact.

| name | class | |
|---|---|---|
| `default` | `DefaultAlgorithm` | vanilla SplitNN |
| `splitfed` | `SplitFedAlgorithm` | SplitFed V2: FedAvg of the client slices each round |
| `splitfed_v1` | `SplitFedV1Algorithm` | SplitFed V1: a server copy per client, both sides averaged |
| — | `LocalLossAlgorithm` | client trains on an auxiliary head, no server gradient |

```python
loaders = ts.data.partition(train_set, 10, "dirichlet", alpha=0.5, seed=0).loaders(train_set, batch_size=32)
session = ts.ClientSession(loaders, mode="parallel")          # or mode="sequential" (handoff)
history = ts.slice(model, n=2).train(session, run_config=cfg, algorithm=ts.SplitFedAlgorithm())
```

From a config: `sl.algorithm: splitfed`, `sl.clients: 10`, `sl.client_mode: parallel`.
Register your own with `ts.register_algorithm`
([tutorial](docs/tutorials/algorithm.md)).

A `ClientSession` keeps one state per client (client-slice weights and optimizer state) and
swaps it in between batches. States live in host RAM or on the GPU (`sl.client_states:
cpu|gpu|auto`), and TorchSlicer checks before training that they fit in memory. Today all
clients are simulated in the data owner's process. Clients as separate nodes with their
own data are planned.

## Data, client splits and evaluation

With a `data:` section, `ts.run(model_fn, run_config=cfg)` needs no `data_fn`. It loads the
dataset, splits it across `sl.clients`, and evaluates on the test set:

```yaml
data:
  dataset: cifar10              # mnist | fashion_mnist | cifar10 | cifar100 | synthetic | npz
  partition: dirichlet          # iid | dirichlet | shards | quantity
  partition_params: {alpha: 0.5}
  batch_size: 32
  seed: 0
  train_subset: 5000            # smoke tests; 0 = all
  eval_every: 1                 # test set every N epochs
sl: {setting: vanilla, clients: 10, client_mode: parallel, algorithm: splitfed}
```

The split is deterministic for a seed and is saved as `runs/<run_id>/client_split.json`.
Add partitioners with `ts.data.register_partitioner`. Evaluation reports `test_loss` and
`test_accuracy` (per client too, for sessions), runs cut hooks as deployed, and leaves
training untouched (the RNG is restored). For now it is supported on the local executor
only. Example: [`examples/data/train.py`](examples/data/train.py) with
[`experiments/m11_cifar10_dirichlet.yaml`](experiments/m11_cifar10_dirichlet.yaml);
[tutorial](docs/tutorials/data.md).

## Privacy: attacks, defences, leakage metrics

`ts.privacy` attacks observe a cut from the side that receives it (after the cut hooks), and
score themselves against the batch's true labels or inputs; defences add a term to the
client's loss (NoPeek), or are cut hooks (noise, quantization, top-k, per-sample clipped
noise). Observing never changes training: the losses are bit-identical with or without
passive attacks. The one active attack, FSHA, is a malicious server that replaces the
returned gradient on purpose.

```yaml
privacy:
  attacks:
    - {name: label_norm, cut: 0}         # leak_label_norm_auc (Li et al., ICLR 2022)
    - {name: inversion, cut: 0}          # leak_inversion_psnr / _ssim, aux = the test set
    - {name: dcor, cut: 0}               # distance correlation, inputs vs activations
    - {name: unsplit, cut: 0}            # architecture-only inversion + model stealing
    - {name: dp, cut: 0}                 # (epsilon, delta) of the clip_noise hook below
  defences:
    - {name: nopeek, cut: 0, params: {alpha: 1.0}}
sl:
  cut_hooks:
    - {cut: 0, name: clip_noise, params: {clip: 0.01, noise_multiplier: 2.0, forward: false, grad: true}}
```

The same from Python: `sliced.train(..., privacy=[ts.privacy.NormLabelAttack(), ts.privacy.NoPeek(alpha=1.0)])`.
It runs on the local executor (the simulator). Benchmark: [`benchmarks/privacy/`](benchmarks/privacy/);
[tutorial](docs/tutorials/privacy.md).

## Transport, NAT and fault tolerance

**Tensor transport** (`transport.tensor`, env `TENSOR_TRANSPORT`) carries activations and
gradients between slices. Control calls always use gRPC.

| mode | use it for |
|---|---|
| `grpc` | anything, including NAT tunnels and relays (default) |
| `tcp` | directly reachable peers; pinned-memory async copies |
| `ipc` | GPU slices on one machine (CUDA IPC; falls back to TCP) |
| `nccl` | one GPU per slice, multi-GPU or RDMA clusters (`torch.distributed` isend/irecv; gloo for CPU tensors) |

Measured on one A100, 2 slices in 2 processes, mean of 3
([`benchmarks/transport/`](benchmarks/transport/)):

| cut per batch | grpc | tcp | ipc |
|---|---|---|---|
| 0.07 MB | 6.8 ms | 3.4 ms | 3.7 ms |
| 2.1 MB | 24.2 ms | 12.4 ms | 5.0 ms |
| 8.4 MB | 61.2 ms | 15.3 ms | 7.9 ms |
| 67 MB | 1411 ms | 95.4 ms | 20.8 ms |

Below ~1 MB per cut the transport hardly matters. NCCL is checked bit-exact on CPU (gloo)
only so far, since real NCCL needs ≥ 2 GPUs. In containers, set `OMP_NUM_THREADS` to the
CPU quota on every node ([why](docs/tutorials/transport.md)).

**NAT traversal.** A data owner behind NAT or a firewall (`transport.nat: true`) opens no
port. It dials each node that has to call it and opens a tunnel there, so gradients,
U-shaped activations, metrics and heartbeats come back on connections it opened. An
optional relay covers peers that can't reach each other:
`python -m torchslicer relay --port 50060`, `transport.relay`. The tunnel costs nothing
next to direct gRPC ([tutorial](docs/tutorials/nat.md)).

**Fault tolerance.** Heartbeats are always on: a dead worker raises `ts.WorkerFailureError`
instead of hanging the run, and workers report runtime errors to the driver. With
`fault_tolerance.enabled: true` the executor re-plans the model across the surviving
workers and resumes the epoch from the last checkpoint.

## Out-of-core training

`ts.OffloadExecutor` trains, on one node, a model whose weights, gradients and optimizer
state don't fit in GPU memory. Slices live in pinned host RAM or in files and are paged into
the GPU one at a time, overlapped with compute. Backward recomputes each slice's forward
with the same RNG state. The losses match training in GPU memory.

```python
sliced = ts.slice(model, strategy="params", n=4,
                  executor=ts.OffloadExecutor(mode="cpu", optimizer="auto", device="cuda"))
```

Or set `offload.mode: cpu|disk` in the config. In one measurement, a 666M-parameter model
under a 6 GB GPU cap took 0.84 s/step, against 0.61 s unsplit with no cap and 5.1 s for
FSDP2 offload with checkpointing ([tutorial](docs/tutorials/offload.md),
[`benchmarks/offload/`](benchmarks/offload/)).

## Configuration

`ts.RunConfig` is a tree of dataclasses: `training`, `pipeline`, `discovery`, `startup`,
`transport`, `checkpoint`, `logging`, `profile`, `fault_tolerance`, `network`, `planning`,
`deploy`, `sl`, `offload`, `data`. Priority: **Python kwargs > env vars > YAML > defaults**,
so a per-node env var beats the shared file.

```python
cfg = ts.RunConfig.load("experiments/resnet18_4gpu.yaml")   # YAML, then env overrides
cfg.training.epochs = 3
sliced.train(loader, run_config=cfg)                         # everything from the config
cfg.to_yaml("runs/my_run/resolved_config.yaml")              # save for reproducibility
```

```yaml
training:
  epochs: 10
  optimizer: {name: AdamW, params: {lr: 3.0e-4}}
  criterion: {name: CrossEntropyLoss, params: {}}
  scheduler: {name: linear, params: {warmup_steps: 500, total_steps: 20000}}   # local executor
  mixed_precision: true          # bf16 autocast
  deterministic: true            # same seed -> same losses on GPU too, in every process
  sync_back: all                 # after a cluster run, the trained weights come back into `model`
pipeline: {use_gpipe: true, n_micro: 4}
discovery: {n_workers: 3}        # number of slices
deploy: {topology: p2p}
transport: {tensor: tcp}
network: {device: auto}
fault_tolerance: {enabled: true}
```

The env var for each field is listed in `_ENV_OVERRIDES` in
[`config.py`](lib/torchslicer/torchslicer/config.py) (and in `.env.example` for Docker).
[`experiments/`](experiments/) has ready-made configs.

## Monitoring

- **Run logs:** with `logging.enabled`, each run writes structured JSONL, the resolved config
  and checkpoints under `runs/<run_id>/`. `ts.restore_model_from_run(...)` rebuilds the
  trained model from them. After `train()` the model you sliced is already trained, on a
  cluster too: the data owner reads the slices back from the workers at the end of the run
  (`training.sync_back`: `all` by default, `hosted` = only its own slices, `none`).
- **Profiling:** per-phase timing and memory, collected from the workers.
- **Callbacks:** subclass `ts.TrainingCallback` for per-batch and per-epoch hooks.
- **Tracing:** OpenTelemetry, one trace per batch with context propagated across workers,
  ready for Arize Phoenix. It's a no-op unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Try
  `make demo-tracing`.

## Docker

```bash
make build-cpu                                            # or build-gpu
make run-centralized CONFIG=experiments/resnet18_4gpu.yaml [DEVICE=gpu]
make run-p2p CONFIG=experiments/resnet18_2gpu_p2p.yaml
make run-workers COORDINATOR_ADDRESS=host:50054           # workers only, coordinator elsewhere
make run-phoenix CONFIG=experiments/resnet18_4gpu.yaml    # + Phoenix tracing UI on :6006
make help                                                 # everything else
```

The dev compose files bind-mount `lib/torchslicer`, so code changes need no image rebuild.

## Repository layout

```
lib/torchslicer/torchslicer/   the package
  core/                        tracing (model_graph), SplitLayer, pack stages
  strategies/                  splitters and the strategy registry
  executors/                   local, distributed (coordinator), worker, p2p
  transport/                   gRPC services, TCP/IPC/NCCL tensor transport, NAT tunnels
  sl/                          roles, cut hooks, metrics, client sessions
  algorithm/                   ts.Algorithm and the built-in SL algorithms
  data/                        datasets, partitioners, build_data
  offload/                     OffloadExecutor and the slice pager
  deploy/                      ts.run, node planning, launcher targets
  monitor/                     run logger, profiler, OpenTelemetry tracer
  config.py                    RunConfig
docs/tutorials/, docs/deploy/  guides (code blocks are tested)
examples/                      entry points: run_anywhere, sl, data, train/{centralized,p2p,lm,hf}, pack, finetune
experiments/                   RunConfig YAML files
benchmarks/                    equivalence matrix, transport, NAT, offload, clients
tests/                         pytest suite
```

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e '.[grpc]' torch torchvision pytest
.venv/bin/python -m pytest tests/
```

After editing a `.proto`, regenerate the stubs (the Docker entrypoint does this too):

```bash
cd lib/torchslicer && python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. \
  torchslicer/transport/grpc/coordinator/coordinator_service.proto \
  torchslicer/transport/grpc/worker/worker_service.proto
```

There are two `pyproject.toml` files: the root one for editable installs, and
`lib/torchslicer/pyproject.toml`, which CI publishes to PyPI on `v*.*.*` tags. Keep their
versions and dependencies in sync, along with `__version__`.

**Known limits:** test-set evaluation isn't available with offload or with SplitFed V1 on
client nodes; channels are unencrypted and
unauthenticated (TLS/mTLS is planned), so run on trusted networks; real multi-GPU NCCL is
not yet validated.

## Citing

If you use TorchSlicer in your research, please cite it. The metadata is in
[`CITATION.cff`](CITATION.cff).

## License

GPL-3.0-or-later
