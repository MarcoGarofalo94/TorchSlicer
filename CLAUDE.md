# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Interaction Rules

- Responses must be extremely concise. No unnecessary caveats, preamble, or filler.
- Do not generate `.md` files unless explicitly requested.
- Git commits must have only the user as author. Do not add co-authorship lines from any tool. Use conventional commits. No phase references in commit messages.
- Current development environment is a WSL2 with an Ubuntu image, equipped with Docker containers.

## Environment

- Conda environment: `torchslicer` (Python 3.12)
- Install library: `conda run -n torchslicer pip install -e lib/torchslicer`
- Run scripts: `conda run -n torchslicer python3 <script>`

## Project Vision

TorchSlicer is a research framework implementing **split learning** in PyTorch. The core idea: partition a neural network into contiguous slices, assign each slice to a device, pass activations forward and gradients backward — without any device ever holding the full model or full dataset.

**Primary motivations:**
- Train large models on constrained devices (even a single device, sequentially)
- Distributed training across heterogeneous (CPU/GPU) nodes
- Research platform for comparing split strategies, topologies, and transports

**Target usage** (add-on over existing models):
```python
import torchslicer as ts
model = torchvision.models.resnet50()
sliced = ts.slice(model, strategy="uniform", n=4)
sliced.train(train_loader, devices=[...])
```

---

## Architecture (Target)

### Layer 1 — Core (`torchslicer/core/`)
- **`Slicer`**: Inspects a `nn.Module`, extracts child layers, serialises each via `torch.save()` for transport. Works for any `nn.Module` — no metadata mining.
- **`SplitLayer`**: Wraps a contiguous partition; owns `forward`, `backward`, `optimize`. DAG-aware: accepts `predecessors` list for intra-partition skip connections.
- **`ModelGraph`**: DAG representation of a model. `from_module()` uses `torch.fx` shallow trace to capture functional ops (`torch.flatten`, `operator.add`) as thin `nn.Module` wrappers.

### Layer 2 — Splitting Strategies (`torchslicer/strategies/`)
- **`BaseSplitter`**: Public abstract base class users subclass to implement custom strategies. Must expose a clean, minimal interface (e.g. `split(model_graph, devices) -> List[Partition]`).
- **`UniformSplitter`**: Equal number of layers per partition (baseline). Populates intra-partition DAG predecessor info.
- **`EnergySplitter`**: Minimize energy consumption given device profiles (not implemented)
- **`DeadlineSplitter`**: Meet latency/deadline constraints (not implemented)
- Strategies must be registerable so users can pass them by name or instance to the top-level API: `ts.slice(model, strategy=MyCustomSplitter())`

### Layer 3 — Execution (`torchslicer/executors/`)
- **`LocalExecutor`**: Single device, sequential execution (fits large model in constrained memory). Supports `verbose=True` for per-batch loss logging.
- **`DistributedExecutor`**: Centralized topology — this process acts as coordinator. Starts an embedded gRPC server, sends `SliceConfig` to remote workers, drives the training loop synchronously via `threading.Event`.

### Layer 4 — Transport (`torchslicer/transport/`)
- **`GRPCTransport`**: Current implementation (gRPC + protobuf). Proto files live in `lib/torchslicer/torchslicer/transport/grpc/`; auto-compiled at container startup by `entrypoint.sh`.
- **`RESTTransport`**: HTTP/REST alternative (not yet implemented)
- **`InProcessTransport`**: For single-machine simulation (not yet implemented)

### Layer 5 — Topology (`torchslicer/topology/`)
- **`CentralizedTopology`**: Coordinator orchestrates training loop, workers execute slices
- **`P2PTopology`**: Workers self-coordinate (not yet implemented)

### Layer 6 — Monitoring (`torchslicer/monitor/`)
- Experiment logging (loss, gradients, timing per slice)
- Device profiling (energy, latency)
- Comparison harness for benchmarking strategies/transports/topologies

---

## Current State

### What works
- `Slicer` (torch.save-based, any nn.Module) + `SplitLayer` (DAG-aware) + `ModelGraph` (torch.fx shallow trace)
- `LocalExecutor`: single-device training with optional verbose logging, intra-partition DAG support
- `DistributedExecutor`: centralized gRPC coordinator embedded in the executor; full Docker stack verified
- `UniformSplitter` + strategy registry
- Top-level `ts.slice()` + `SlicedModel.train()` API
- ResNet18/50 work natively via `from_module()` — `torch.flatten` auto-wrapped, no manual wrapper needed
- gRPC centralized example: coordinator + 2–4 workers, automatic partitioning via `UniformSplitter`
- GPipe micro-batch pipeline parallelism — opt-in via `use_gpipe=True, n_micro_batches=4`; benchmarked 1.9× speedup with 4 workers (ResNet18/CIFAR-10 GPU)
- `N_WORKERS`, `EPOCHS`, `USE_GPIPE`, `N_MICRO` env vars — configure stack from outside without rebuilding
- Docker stack: `docker compose up` (CPU) or `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up` (GPU)
- Tested on: RTX 3060 (WSL2 + NVIDIA Container Toolkit)

### What is incomplete
- No P2P topology
- No REST transport logic (Dockerfiles ready, logic absent)
- No monitoring or experiment logging
- No `EnergySplitter` / `DeadlineSplitter`
- Cross-partition skip connections not supported (intra-partition DAG works; cross-partition requires protocol changes)
- `DistributedExecutor` workers receive flat sequential layer lists (intra-partition DAG not transmitted over wire)

---

## Common Commands

### Run the local DNN smoke test (no Docker needed)
```bash
conda run -n torchslicer python3 examples/test_local_dnn.py
```

### Run the full gRPC stack (CPU)
```bash
docker compose up --build
```
Starts coordinator (port 50054) + worker1–4 (port 50051) on a bridged Docker network.
Source and lib are volume-mounted; proto files are regenerated at container startup via `entrypoint.sh`.

Environment variables (pass via shell or `.env`):
```bash
N_WORKERS=4 EPOCHS=20 USE_GPIPE=1 N_MICRO=4 docker compose up
```

### Run the full gRPC stack (GPU)
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```
Requires NVIDIA Container Toolkit. Workers get GPU access; coordinator runs on CPU.

### Build images manually
```bash
make help          # list all targets
make build-cpu     # build CPU image
make build-gpu     # build GPU image
```

### Regenerate protobuf files (run from repo root)
```bash
conda run -n torchslicer python3 -m grpc_tools.protoc \
    -I lib/torchslicer \
    --python_out=lib/torchslicer \
    --grpc_python_out=lib/torchslicer \
    torchslicer/transport/grpc/coordinator/coordinator_service.proto \
    torchslicer/transport/grpc/worker/worker_service.proto
```
Note: proto files are also auto-regenerated at container startup by `entrypoint.sh`.

### Run directly (outside Docker)
```bash
# From examples/train/centralized/GRPC/
python3 coordinator/main.py 50054
python3 worker/main.py 50051
```

---

## Docker Image Structure

Two images, both role-agnostic (coordinator and worker run from the same image):

| Image | Base | Contents |
|-------|------|----------|
| `Dockerfile.cpu` | `python:3.12-slim` | torch-cpu + grpcio + fastapi + torchslicer |
| `Dockerfile.gpu` | `python:3.12-slim` | torch-cu121 + grpcio + fastapi + torchslicer |

- `entrypoint.sh`: recompiles proto files from `/usr/local/lib/torchslicer` at startup, ensuring gencode/runtime version always match.
- `docker-compose.gpu.yml`: override file that switches all services to the GPU image and adds NVIDIA device reservations to workers.
- Role (coordinator vs worker) is selected via `command:` in docker-compose, not by separate images.

---

## gRPC Protocol Design

### `coordinator_service.proto` (package `torchslicer.coordinator`)
- `batch_done(BatchDoneRequest)` — first worker signals batch completion; carries `batch_id`
- `report_metrics(MetricsMessage)` — last worker reports `loss`, `batch_id`, `worker` hostname

### `worker_service.proto` (package `torchslicer.worker`)
- `init(SliceConfig)` — send layers + optimizer + criterion to a worker at startup
- `forward(ForwardRequest)` — carry `batch_id` + input tensor (or label for the last worker)
- `backward(BackwardRequest)` — carry `batch_id` + gradient tensor

Key design points:
- **`batch_id`** on every message — required for correct gradient↔forward matching and for future pipeline parallelism
- **Label travels in `ForwardRequest`** (not a separate `set_label` RPC) — coordinator sends `ForwardRequest(batch_id, label)` directly to the last worker, eliminating the label/activation race
- **`LayerConfig`** holds `layer_type` (for logging) + `serialized` bytes (`torch.save(layer)`) — no JSON blobs, no attribute mining
- **`Tensor`** carries `shape` and `DType` enum alongside `data` bytes — enables validation and future mixed-precision without full deserialisation

---

## Key Design Decisions

- **Layer serialisation**: `torch.save(layer)` per layer captures architecture + weights in one shot. Works for any `nn.Module`; no inspect/regex/eval. Constraint: the layer class must be importable on the worker.
- **ModelGraph tracing**: `from_module()` uses `torch.fx` with a `_ShallowTracer` that treats direct children as leaves. Functional ops (`torch.flatten`, `operator.add`) are wrapped in thin `nn.Module`s. Falls back to `from_sequential()` on trace failure.
- **Cross-partition skip connections**: Not supported. `BaseSplitter.validate()` raises `ValueError` if a multi-input node's predecessors span partitions. Workaround: choose `n` so skip connections stay within one partition, or wrap the skip block in a single `nn.Module`.
- **Tensor serialisation**: `torch.save()` → `bytes`. Simple but not the most efficient; revisit for optimization.
- **Gradient flow**: Each `SplitLayer` stores its input with `detach().requires_grad_(True)` to enable `x.grad` retrieval after backward. For GPipe, `x_ref = self.layer.x` is captured immediately after each forward and keyed by batch_id to avoid overwrite across micro-batches.
- **Hybrid clusters**: For GPU workers, `torch.save(layer)` serialises on CPU; the worker can `.to(device)` after loading. Device placement is the worker's responsibility.
