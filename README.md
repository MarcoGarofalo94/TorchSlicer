<p align="center">
  <img src="examples/monitor/static/torchslicer_logo.svg" alt="TorchSlicer logo" width="140" />
</p>

<h1 align="center">TorchSlicer</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-black" alt="MIT License" /></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/runtime-local%20%7C%20centralized%20%7C%20p2p-1f6feb" alt="Supported runtimes" />
  <img src="https://img.shields.io/badge/transport-grpc%20%7C%20tcp-0f766e" alt="Supported transports" />
</p>

<p align="center">
  Split learning for PyTorch, with local execution, centralized orchestration, P2P runtime support, and explicit stage packing for models that need manual partitioning.
</p>

TorchSlicer partitions a model into stages, places those stages across devices or hosts, and runs forward activations and backward gradients across the resulting pipeline.

Today the project supports:
- local sliced execution
- centralized distributed execution with a coordinator
- peer-to-peer execution without a separate coordinator process
- explicit stage packing for models that are not easy to slice with default tracing
- checkpoint reconstruction utilities
- configurable worker-to-worker tensor transport for distributed execution

TorchSlicer is designed for experimentation with model partitioning strategies and distributed training layouts, while keeping the core runtime reusable and library-focused.

## At a Glance

- Core library for split learning in PyTorch
- Local, centralized, and P2P execution modes
- Automatic slicing and explicit `pack(model)` stage definitions
- YAML, env, and Python-driven runtime configuration
- Distributed checkpoint reconstruction utilities
- Configurable worker-to-worker tensor transport

## What It Solves

Split learning is useful when:
- a single device cannot fit the full model
- you want to distribute training across consumer-grade devices
- you want to experiment with where a network should be partitioned
- you need more control than a monolithic data-parallel setup provides

TorchSlicer focuses on model partitioning and stage-to-stage execution. It does not try to hide that split learning has real tradeoffs: communication cost, stage imbalance, and orchestration overhead matter, and the best split depends on the model and hardware.

## Current Capabilities

- Slice a model automatically with a splitting strategy such as `uniform`
- Provide a custom `pack(model)` function when tracing is not enough
- Train with a local executor for fast iteration
- Train with a centralized distributed executor across workers
- Train with a P2P topology for coordinator-free runs
- Configure runs through Python, YAML, or environment variables
- Collect run logs, profiling metrics, and checkpoints
- Reconstruct a model from saved distributed checkpoints

## Installation

Python 3.10+ is required.

Install the package in editable mode:

```bash
python -m pip install -e lib/torchslicer
```

If you need distributed gRPC execution support:

```bash
python -m pip install -e 'lib/torchslicer[grpc]'
```

If you want tracing/monitoring extras:

```bash
python -m pip install -e 'lib/torchslicer[monitor]'
```

For PEFT/Hugging Face adapter workflows:

```bash
python -m pip install -e 'lib/torchslicer[peft]'
```

## Quick Start

### 1. Slice a model

```python
import torch.nn as nn
import torchslicer as ts

model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

sliced = ts.slice(model, strategy="uniform", n=2)
```

By default, `ts.slice(...)` uses a local executor. That is useful for validating partitions before moving to a distributed setup.

### 2. Train locally

```python
history = sliced.train(
    data_loader=train_loader,
    optimizer={"name": "SGD", "params": {"lr": 0.01, "momentum": 0.9}},
    criterion={"name": "CrossEntropyLoss", "params": {}},
    epochs=5,
)
```

### 3. Use a distributed executor

```python
import torchslicer as ts

executor = ts.DistributedExecutor()
sliced = ts.slice(model, strategy="uniform", n=2, executor=executor)
```

Then start the coordinator and worker processes with the provided examples or Docker Compose flows.

## Packed Stages

Some models are better described explicitly as stages than discovered automatically through tracing. For those cases, TorchSlicer accepts a `pack(model)` callable that returns an ordered list of `nn.Module` stages.

```python
import torchslicer as ts

def pack_model(model):
    return [
        ts.BlockStage(model.embed),
        *[ts.BlockStage(block) for block in model.blocks],
        ts.BlockStage(model.head),
    ]

sliced = ts.slice(model, n=4, pack=pack_model)
```

Important constraint:
- every class used inside `pack(model)` must be importable on worker processes under the same module path

## Topologies

### Centralized

A coordinator initializes workers, assigns slices, injects inputs and labels, and tracks run state. Worker-to-worker execution still happens stage by stage across the selected transport.

Run with Docker Compose:

```bash
make run-centralized CONFIG=experiments/resnet18_4gpu.yaml
```

GPU variant:

```bash
make run-centralized DEVICE=gpu CONFIG=experiments/resnet18_4gpu.yaml
```

### P2P

P2P mode removes the separate coordinator process. One worker acts as the driver for the training run and the other workers connect as peers.

```bash
make run-p2p CONFIG=experiments/resnet18_2gpu_p2p.yaml
```

GPU variant:

```bash
make run-p2p DEVICE=gpu CONFIG=experiments/resnet18_2gpu_p2p.yaml
```

## Runtime Configuration

TorchSlicer uses a single `RunConfig` model with the following precedence:

`Python kwargs > YAML > environment variables > defaults`

That means you can:
- configure runs directly from Python
- keep reusable experiment settings in `experiments/*.yaml`
- override selected values from Docker Compose or shell env vars

Key configuration areas include:
- training
- pipeline and micro-batching
- discovery
- startup and retries
- checkpointing
- logging and profiling
- transport
- fault tolerance

## Transport

Distributed execution now supports configurable worker-to-worker tensor transport.

Default:
- `grpc` tensor transport

Optional:
- `tcp` tensor transport for lower-overhead worker-to-worker tensor payload exchange

This is configured through `RunConfig`:

```yaml
transport:
  tensor: tcp
  tensor_port_offset: 1
```

The default behavior remains unchanged unless you opt into the TCP path.

## Examples and Entry Points

Useful starting points in this repository:
- [examples/train/centralized/GRPC/coordinator/main.py](/home/marcow10/TorchSlicer/examples/train/centralized/GRPC/coordinator/main.py)
- [examples/train/centralized/GRPC/worker/main.py](/home/marcow10/TorchSlicer/examples/train/centralized/GRPC/worker/main.py)
- [examples/train/p2p/worker/main.py](/home/marcow10/TorchSlicer/examples/train/p2p/worker/main.py)
- [examples/test_local_dnn.py](/home/marcow10/TorchSlicer/examples/test_local_dnn.py)

Useful commands:
- `python -m pip install -e lib/torchslicer`
- `pytest`
- `make help`
- `make build-cpu`
- `make build-gpu`

## Checkpoints and Recovery

TorchSlicer includes utilities to restore a full model from saved distributed checkpoints.

Relevant API:
- `torchslicer.restore_model_from_run(...)`
- `torchslicer.resolve_checkpoint_epoch(...)`

This is useful when training was performed on partitioned workers but you want to reconstruct the model afterward for evaluation or export.

## Project Layout

- `lib/torchslicer/torchslicer/core/`: slicing primitives, graph logic, packed stages
- `lib/torchslicer/torchslicer/executors/`: local and distributed runtimes
- `lib/torchslicer/torchslicer/discovery/`: worker registration and peer resolution
- `lib/torchslicer/torchslicer/transport/`: transport abstractions and gRPC bindings
- `lib/torchslicer/torchslicer/strategies/`: splitting strategies
- `examples/`: runnable entry points and reference scripts
- `experiments/`: YAML configurations
- `tests/`: unit and regression tests
- `docker-images/`: container images for CPU and GPU execution

## Development

Run the test suite:

```bash
pytest
```

Build images:

```bash
make build-cpu
make build-gpu
```

Show available commands:

```bash
make help
```

## Limitations

Current limitations to be aware of:
- split quality still depends heavily on partition choice
- communication overhead can dominate on small models or small batches
- some advanced strategies are still incomplete
- not every architecture is automatically traceable, so `pack(model)` is sometimes required
- transport and topology improvements are still evolving

## Status

TorchSlicer is already usable for split-learning experiments and runtime development, but it is still evolving. The public entrypoints are stable enough to build on, while the runtime and transport layers continue to improve.

If you want to contribute, the best starting points are:
- executor/runtime improvements
- new partitioning strategies
- better examples and docs
- regression tests around distributed execution
