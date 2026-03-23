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
- Distributed training across heterogeneous (CPU/GPU) nodes in a real network
- Research platform for comparing split strategies, topologies, and transports

**Design intent**: devices join the cluster by running the worker process (container or host), registering with the coordinator. The coordinator discovers them dynamically — no hardcoded hostnames, no manual address configuration.

**Target usage** (add-on over existing models):
```python
import torchslicer as ts
model = torchvision.models.resnet50()
sliced = ts.slice(model, strategy="uniform", n=4)
sliced.train(train_loader, devices=[...])
```

---

## Architecture

### Layer 1 — Core (`torchslicer/core/`)
- **`Slicer`**: Inspects a `nn.Module`, extracts child layers, serialises each via `torch.save()` for transport. Works for any `nn.Module` — no metadata mining.
- **`SplitLayer`**: Wraps a contiguous partition; owns `forward`, `backward`, `optimize`. DAG-aware: accepts `predecessors` list for intra-partition skip connections.
- **`ModelGraph`**: DAG representation of a model. `from_module()` uses `torch.fx` shallow trace to capture functional ops (`torch.flatten`, `operator.add`) as thin `nn.Module` wrappers.

### Layer 2 — Splitting Strategies (`torchslicer/strategies/`)
- **`BaseSplitter`**: Public abstract base class users subclass to implement custom strategies.
- **`UniformSplitter`**: Equal number of layers per partition (baseline). Populates intra-partition DAG predecessor info.
- **`EnergySplitter`**: Minimize energy consumption given device profiles (not implemented)
- **`DeadlineSplitter`**: Meet latency/deadline constraints (not implemented)
- Strategies are registerable: `ts.slice(model, strategy="uniform")` or by instance/class.

### Layer 3 — Execution (`torchslicer/executors/`)
- **`LocalExecutor`**: Single device, sequential execution. Supports `verbose=True` and GPipe micro-batching.
- **`DistributedExecutor`**: Centralized topology. Embeds a gRPC server, uses a `BaseDiscovery` instance to find workers, sends `SliceConfig`, drives the training loop, and signals clean shutdown via `Shutdown` RPC on teardown.
- **`WorkerServicer`**: gRPC `WorkerService` implementation shared by centralized and P2P workers. Handles `init`, `forward`, `backward`, `shutdown`, `get_stats` RPCs. Lives in `executors/worker.py`; both topology examples import from here.

### Layer 4 — Discovery (`torchslicer/discovery/`)
New layer. Abstracts cluster membership for both centralized and P2P topologies.
- **`BaseDiscovery`**: Abstract interface — `announce()`, `discover()`, `watch()`. `watch()` is a forward-compat hook for fault tolerance (heartbeat callbacks).
- **`CoordinatorDiscovery`**: Workers push a `Register` RPC to the coordinator at startup. Coordinator blocks in `discover()` until N workers have registered. Used for centralized topology.
- **`StaticDiscovery`**: Peer list from config (`peers: [host:port, ...]`). No registration needed. Used for P2P with fixed addresses or testing.
- **`announce_to_coordinator(coordinator_addr, node_info)`**: Client-side helper used by workers; retries until coordinator is reachable.

### Layer 5 — Configuration (`torchslicer/config.py`)
- **`RunConfig`**: Single source of truth. Sub-configs: `TrainingConfig`, `PipelineConfig`, `DiscoveryConfig`, `CheckpointConfig`, `LoggingConfig`, `ProfileConfig`.
- Load priority: Python API kwargs > YAML file > env vars > defaults.
- `RunConfig.load(path)`: merged load — YAML base + env var overrides on top.
- `EXPERIMENT_CONFIG` env var: path to YAML; Docker Compose passes it through automatically.
- **`LoggingConfig`**: `enabled` (default `True`), `dir` (default `./runs`). Env vars: `LOG_ENABLED`, `LOG_DIR`.
- **`ProfileConfig`**: `verbosity` (0–3, default 0), `memory` (bool, default False). Env vars: `PROFILE_VERBOSITY`, `PROFILE_MEMORY`.

### Layer 6 — Transport (`torchslicer/transport/`)
- **`GRPCTransport`**: Current implementation. Proto files in `lib/torchslicer/torchslicer/transport/grpc/`; auto-compiled at container startup by `entrypoint.sh`.
- **`RESTTransport`**: Not yet implemented (Dockerfiles ready).

### Layer 7 — Topology (`torchslicer/topology/`)
- **`CentralizedTopology`**: Coordinator orchestrates training loop, workers execute slices.
- **`P2PTopology`**: Implemented. No coordinator process. Driver node (worker 0) owns the DataLoader, builds the full model, slices it, and sends partitions to followers via `init()` RPC. Embeds `_P2PCoordinatorServicer` to handle `batch_done` + `report_metrics` from followers. Labels sent directly from driver to last worker — intermediate workers never see labels (privacy-preserving). Uses `StaticDiscovery` (peers from config). Entry point: `examples/train/p2p/worker/main.py`; all workers run the same script, role determined by `IS_DRIVER` env var.

### Layer 8 — Monitoring (`torchslicer/monitor/`)
- OpenTelemetry tracer with `configure()`, `span()` context manager.
- `LocalExecutor` and `DistributedExecutor` emit spans for batch/forward/backward with timing and memory attributes.
- Dashboard: FastAPI backend + React/Recharts frontend. Jaeger for trace storage.
- **`RunLogger`**: Per-run artifact writer and reader. Writes per-phase JSONL files to `{logging.dir}/{run_id}/` — each file is schema-homogeneous and directly loadable with pandas. All checkpoints also land in this directory when enabled. `RunLogger.load(run_dir)` + `to_dataframe(phase)` for plotting.
- **`TrainingCallback`**: Base class with no-op defaults (`on_train_begin`, `on_epoch_begin`, `on_epoch_end`, `on_batch_end`, `on_train_end`). `on_epoch_end` receives and returns the metrics dict — custom keys injected here are written to `metrics.jsonl`.
- **`WorkerProfiler`**: Worker-side per-phase timer. Phases: `forward`, `backward`, `optimizer`, `send_fwd`, `send_bwd`, `idle_fwd`, `idle_bwd`. Verbosity 0 = zero overhead. Memory snapshots optional (`profile_memory=True`). Coordinator pulls stats via `get_stats` RPC after each epoch.
- **Per-phase log files** (each homogeneous, load with `pd.read_json(path, lines=True)`):
  - `metrics.jsonl` — epoch loss + duration
  - `coordinator.jsonl` — coordinator overhead (data_load, send, wait) per epoch
  - `worker_epoch.jsonl` — per-worker epoch aggregates (avg/min/max/p95/total per phase + GPU memory)
  - `worker_batch.jsonl` — per-worker per-batch detail (verbosity=3 only)
  - `partition_epoch.jsonl` / `partition_batch.jsonl` — local executor equivalents

---

## Current State

### What works
- `Slicer` + `SplitLayer` (DAG-aware) + `ModelGraph` (torch.fx shallow trace)
- `LocalExecutor`: single-device training, verbose logging, GPipe micro-batching
- `DistributedExecutor`: centralized gRPC topology, embedded coordinator server, dynamic worker discovery
- `CoordinatorDiscovery`: workers self-register at startup; coordinator waits for N registrations; no hardcoded hostnames; any device can join by running the worker process
- `StaticDiscovery`: peer list from config, no registration
- `UniformSplitter` + strategy registry
- Top-level `ts.slice()` + `SlicedModel.train()` API
- ResNet18/50 work natively via `from_module()`
- Intra-partition DAG wired over gRPC: `SliceConfig` carries `PredecessorList` per layer; workers reconstruct and pass to `SplitLayer` — multi-input (skip connection) layers execute correctly in distributed mode
- GPipe micro-batch pipeline parallelism — 1.9× speedup with 4 workers (ResNet18/CIFAR-10 GPU)
- Clean lifecycle: coordinator sends `Shutdown` RPC to all workers on teardown; workers exit cleanly (code 0); coordinator then blocks on `signal.pause()` — only exits on `SIGTERM` (`docker compose down`); idle workers (registered but not selected) also receive `Shutdown` at teardown via `CoordinatorDiscovery.idle_nodes()`
- **P2P topology**: no coordinator process; driver (worker 0) builds model, sends `SliceConfig` to followers, embeds lightweight coordinator service, drives training loop; labels routed directly to last worker (split-learning privacy); followers reuse `WorkerServicer` unchanged; verified: 20 epochs ResNet18/CIFAR-10, 2 GPU workers, both exit 0 (loss 2.4 → 0.52)
- `WorkerServicer` in library (`executors/worker.py`) — shared by centralized and P2P; centralized worker `main.py` now imports from library
- Worker state reset on `init()` — workers reusable across runs without container restart
- Optional checkpoint: each worker saves `{run_dir}/worker_{i}_epoch_{n}.pt`; coordinator saves `run_state.json`; resume via `checkpoint_path` in `SliceConfig`; disabled by default
- `run_id` on all proto messages — forward-compat hook for coordinator restart detection
- `RunConfig` with YAML + env var loading; `.env.example`; `experiments/` directory with ready-to-use configs
- **Run logging system**: always-on by default (`logging.enabled=True`); every run writes `runs/{run_id}/run_manifest.json` + per-phase JSONL files; checkpoints co-located in the same directory when enabled; files owned by host user (UID/GID written to `.env` by `make _env`)
- **Per-phase profiling**: `profile.verbosity` controls granularity (0=off, 1=epoch totals, 2=phase breakdown, 3=per-batch); workers instrument forward/backward/optimizer/send/idle phases; coordinator pulls via new `get_stats` RPC after each epoch
- **Callback system**: `SlicedModel.train(callbacks=[...])` accepts `TrainingCallback` subclasses; `on_epoch_end` can inject custom metrics (accuracy, lr, etc.) into `metrics.jsonl`
- `COORDINATOR_ADDRESS` env var on workers; `WORKER_ADDRESS` for custom address advertisement
- `make run-cpu/run-gpu CONFIG=experiments/resnet18_4gpu.yaml` — one-command experiment launch
- `make run-p2p-gpu/run-p2p-cpu CONFIG=experiments/resnet18_2gpu_p2p.yaml` — P2P stack (no coordinator)
- Docker stack verified: CPU (2–4 workers) and GPU (RTX 3060, WSL2 + NVIDIA Container Toolkit); P2P verified same

### What is incomplete
- No REST transport logic (Dockerfiles ready)
- No fault tolerance: crash aborts the run; `BaseDiscovery.watch()` hook is in place for future heartbeat-based detection; `discover()` raises `TimeoutError` on timeout (intentional, deferred to fault tolerance work)
- No `EnergySplitter` / `DeadlineSplitter`
- Cross-partition skip connections not supported (intra-partition DAG sent over wire and works end-to-end; cross-partition requires protocol changes)
- Device profiling, gradient norm logging, comparison harness not yet implemented

---

## Common Commands

### Run the local DNN smoke test (no Docker needed)
```bash
conda run -n torchslicer python3 examples/test_local_dnn.py
```

### Run the P2P stack (no coordinator)
```bash
# GPU, 2 workers
make run-p2p-gpu CONFIG=experiments/resnet18_2gpu_p2p.yaml

# CPU, 2 workers
make run-p2p-cpu CONFIG=experiments/resnet18_2gpu_p2p.yaml
```

All workers run the same script. `IS_DRIVER=true` on worker0, `IS_DRIVER=false` on all others. `WORKER_PEERS` is a comma-separated peer list in slice-assignment order. Driver retries `init()` until followers are ready (60s timeout).

### Run the full gRPC stack (centralized)
```bash
# CPU, using env vars
N_WORKERS=2 EPOCHS=10 make run-cpu

# CPU, using a YAML experiment config
make run-cpu CONFIG=experiments/resnet18_2cpu.yaml

# GPU, using a YAML experiment config
make run-gpu CONFIG=experiments/resnet18_4gpu.yaml

# GPU + monitoring dashboard
make run-gpu-monitor CONFIG=experiments/resnet18_4gpu.yaml
```

After training the coordinator blocks on `signal.pause()` and only exits on `SIGTERM`. Use `make down` to stop the full stack. Workers stay up and accept the next run without restart.

### Run workers outside Docker (any device on the network)
```bash
# On each worker machine (or container):
COORDINATOR_ADDRESS=<coordinator-ip>:50054 conda run -n torchslicer \
    python3 examples/train/centralized/GRPC/worker/main.py 50051

# On the coordinator machine:
conda run -n torchslicer \
    python3 examples/train/centralized/GRPC/coordinator/main.py 50054 \
    --config experiments/resnet18_4gpu.yaml
```
Workers register with the coordinator via `Register` RPC on startup. The coordinator waits for `N_WORKERS` registrations before dispatching model slices. Workers can run in Docker containers, bare-metal, or any mix.

### Configure an experiment
Three equivalent ways (highest priority first):
```python
# 1. Python API (one-shot tests)
sliced.train(loader, optimizer_cfg, criterion_cfg, epochs=20, use_gpipe=True)

# 2. YAML file (reproducible experiments)
make run-gpu CONFIG=experiments/resnet18_4gpu.yaml

# 3. .env file (persistent defaults for docker compose)
cp .env.example .env  # edit .env
make run-gpu           # writes UID/GID to .env then runs compose
```

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
Proto files are also auto-regenerated at container startup by `entrypoint.sh`.

---

## Docker Image Structure

Two images, both role-agnostic (coordinator and worker run from the same image):

| Image | Base | Contents |
|-------|------|----------|
| `Dockerfile.cpu` | `python:3.12-slim` | torch-cpu + grpcio + fastapi + torchslicer |
| `Dockerfile.gpu` | `python:3.12-slim` | torch-cu121 + grpcio + fastapi + torchslicer |

- `entrypoint.sh`: recompiles proto files from `/usr/local/lib/torchslicer` at startup.
- `docker-compose.gpu.yml`: override file — switches to GPU image, adds NVIDIA device reservations on workers.
- `docker-compose.monitor.yml`: overlay — adds Jaeger (port 16686) + dashboard (port 8080).
- Role (coordinator vs worker) is selected via `command:` in docker-compose, not by separate images.
- Workers get `COORDINATOR_ADDRESS=coordinator:50054` from compose; override for non-Docker deployments.

---

## gRPC Protocol Design

### `coordinator_service.proto` (package `torchslicer.coordinator`)
- `register(RegisterRequest)` — worker announces itself at startup; coordinator assigns `run_id` and `worker_index`
- `batch_done(BatchDoneRequest)` — first worker signals batch completion; carries `batch_id`, `run_id`
- `report_metrics(MetricsMessage)` — last worker reports `loss`, `batch_id`, `worker`, `run_id`

### `worker_service.proto` (package `torchslicer.worker`)
- `init(SliceConfig)` — send layers + optimizer + criterion; includes `run_id`, `worker_index`, `checkpoint_path`, `profile_verbosity`, `profile_memory`, `predecessors` (repeated `PredecessorList` — partition-local DAG edges, one per layer)
- `forward(ForwardRequest)` — carry `batch_id` + input tensor (or label for the last worker)
- `backward(BackwardRequest)` — carry `batch_id` + gradient tensor
- `shutdown(ShutdownRequest)` — graceful stop; `save_checkpoint` flag triggers slice save before `server.stop()`
- `get_stats(GetStatsRequest)` — coordinator pulls per-phase profiling stats after each epoch; returns `WorkerStatsResponse` with `PhaseStats` per phase and optional `BatchStats` list (verbosity=3)

Key design points:
- **`run_id`** on all messages — coordinator-assigned ID; forward-compat hook for coordinator restart detection and future re-registration
- **`worker_index`** in `RegisterResponse` and `SliceConfig` — determines slice assignment order; decoupled from hostname
- **`batch_id`** on every batch message — required for gradient↔forward matching and GPipe pipeline parallelism
- **Label travels in `ForwardRequest`** — coordinator sends label directly to last worker, eliminating label/activation race
- **`LayerConfig`** holds `layer_type` + `serialized` bytes (`torch.save(layer)`) — no JSON blobs, no attribute mining
- **`Tensor`** carries `shape` and `DType` enum — enables validation and future mixed-precision
- **`NodeInfo`** in `RegisterRequest` — carries `node_id`, `address`, `device`, `memory_mb`; used for logging and future splitter strategies

---

## Key Design Decisions

- **Layer serialisation**: `torch.save(layer)` per layer captures architecture + weights in one shot. Constraint: the layer class must be importable on the worker.
- **ModelGraph tracing**: `from_module()` uses `torch.fx` with a `_ShallowTracer` that treats direct children as leaves. Functional ops auto-wrapped. Falls back to `from_sequential()` on trace failure.
- **Dynamic discovery**: Workers register themselves. Coordinator never has hardcoded addresses. Any process (container or bare-metal) that reaches the coordinator's `Register` endpoint can join the cluster.
- **Checkpoint design**: Distributed by nature — each worker saves its own slice independently. No full-model reassembly at checkpoint time. Coordinator saves `run_state.json` for resume metadata.
- **Cross-partition skip connections**: Not supported. `BaseSplitter.validate()` raises `ValueError` if a multi-input node's predecessors span partitions. Workaround: choose `n` so skip connections stay within one partition, or wrap the block in a single `nn.Module`.
- **Gradient flow**: Each `SplitLayer` stores its input with `detach().requires_grad_(True)`. For GPipe, `x_ref = self.layer.x` is captured immediately after each forward and keyed by `batch_id`.
- **Hybrid clusters**: `torch.save(layer)` serialises on CPU; workers call `.to(device)` after loading. Device placement is the worker's responsibility, advertised via `NodeInfo.device` at registration.
- **Fault tolerance (future)**: `BaseDiscovery.watch(on_join, on_leave)` is the hook point. A heartbeat mechanism will fire `on_leave(node)` on failure, letting the executor decide how to handle it without any interface refactor.
