# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Interaction Rules

- Responses must be extremely concise. No unnecessary caveats, preamble, or filler.
- Do not generate `.md` files unless explicitly requested.
- Git commits must have only the user as author. Do not add co-authorship lines from any tool. Use conventional commits.
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
- **`SplitLayer`**: Wraps a contiguous partition; owns `forward`, `backward`, `optimize`
- **`ModelGraph`**: DAG representation of a model; needed for non-sequential models (ResNet skip connections)

### Layer 2 — Splitting Strategies (`torchslicer/strategies/`)
- **`BaseSplitter`**: Public abstract base class users subclass to implement custom strategies. Must expose a clean, minimal interface (e.g. `split(model_graph, devices) -> List[Partition]`).
- **`UniformSplitter`**: Equal number of layers per partition (baseline)
- **`EnergySplitter`**: Minimize energy consumption given device profiles
- **`DeadlineSplitter`**: Meet latency/deadline constraints
- Strategies must be registerable so users can pass them by name or instance to the top-level API: `ts.slice(model, strategy=MyCustomSplitter())`

### Layer 3 — Execution (`torchslicer/executors/`)
- **`LocalExecutor`**: Single device, sequential execution (fits large model in constrained memory). Supports `verbose=True` for per-batch loss logging.
- **`DistributedExecutor`**: Multi-node execution, delegates to Transport (not yet implemented)

### Layer 4 — Transport (`torchslicer/transport/`)
- **`GRPCTransport`**: Current implementation (gRPC + protobuf)
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
- `Slicer` (torch.save-based, any nn.Module) + `SplitLayer` + `ModelGraph` (sequential only)
- `LocalExecutor`: single-device training with optional verbose logging
- `UniformSplitter` + strategy registry
- Top-level `ts.slice()` + `SlicedModel.train()` API
- gRPC centralized example: coordinator + 2 workers, automatic partitioning via `UniformSplitter`
- Docker stack: `docker compose up` (CPU) or `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up` (GPU)
- Tested on: RTX 3060 (WSL2 + NVIDIA Container Toolkit)

### What is incomplete
- No P2P topology
- No REST transport logic (Dockerfiles ready, logic absent)
- No non-sequential model support (skip connections / DAG)
- No monitoring or experiment logging
- No `DistributedExecutor` (gRPC example bypasses library executors)
- No `EnergySplitter` / `DeadlineSplitter`

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
Starts coordinator (port 50054) + worker1 + worker2 (port 50051) on a bridged Docker network.
Source and lib are volume-mounted; proto files are regenerated at container startup via `entrypoint.sh`.

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

### Regenerate protobuf files (run from `examples/train/centralized/GRPC/`)
```bash
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. \
    proto_common/coordinator/coordinator_service.proto \
    proto_common/worker/worker_service.proto
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

- `entrypoint.sh`: scans `/workspace` for `.proto` files and recompiles them at startup, ensuring gencode/runtime version always match.
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

## Development Phases (Roadmap)

### Phase 1 — Refactor & clean API ✅
Restructured `lib/torchslicer` into the layered architecture. Established `BaseSplitter`, `BaseExecutor`, `BaseTransport` interfaces. Cleaned Docker images, gRPC proto schema, Slicer.

### Phase 2 — Local executor ✅
`LocalExecutor` for single-device sequential training, validates the core API without networking.

### Phase 3 — Splitting strategies
Implement `UniformSplitter` (done), then `EnergySplitter` and `DeadlineSplitter`.

### Phase 4 — P2P topology
Implement `P2PTopology` alongside `CentralizedTopology`.

### Phase 5 — REST transport
Implement `RESTTransport` to match `GRPCTransport`.

### Phase 6 — Monitoring & benchmarking
Experiment logging, device profiling, comparison harness. Enable reproducible comparisons across strategies/transports/topologies.

### Phase 7 — Optimization
Mixed-precision, async pipelining, more efficient gradient propagation beyond basic autograd.

---

## Key Design Decisions

- **Layer serialisation**: `torch.save(layer)` per layer captures architecture + weights in one shot. Works for any `nn.Module`; no inspect/regex/eval. Constraint: the layer class must be importable on the worker.
- **Non-sequential models**: Must build a `ModelGraph` (DAG) rather than a flat list of layers. Skip connections require sending activations to non-adjacent slices. Not yet implemented.
- **Tensor serialisation**: `torch.save()` → `bytes`. Simple but not the most efficient; revisit in Phase 7.
- **Gradient flow**: Each `SplitLayer` stores its input as a `Variable(requires_grad=True)` to enable `x.grad` retrieval after backward.
- **Hybrid clusters**: For GPU workers, `torch.save(layer)` serialises on CPU; the worker can `.to(device)` after loading. Device placement is the worker's responsibility.
