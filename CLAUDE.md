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

### Layer 8 — HF Adapter (`torchslicer/adapters/`)
- **`HFAdapter(nn.Sequential)`**: wraps a HuggingFace model as a flat sequence of opaque stage modules, bypassing cross-partition residual connections entirely. Each stage is a single `nn.Module` leaf to ModelGraph's shallow tracer.
- **`wrap_hf(model, task)`**: constructs `HFAdapter` from any HF model. Architecture detection order: user registry → GPT-2 → BERT → LLaMA/Mistral/Qwen/DeepSeek → generic fallback. Supports `causal_lm`, `seq_cls`, `masked_lm` tasks.
- **`register_hf_architecture(name, detect, extract)`**: plug in any architecture (LLaVA, InternVL, Florence-2, …) without forking the library. `detect(model) -> bool` selects the extractor; `extract(model, task) -> list[nn.Module]` returns the flat stage list. Last registered wins.
- **Public stage classes** (`ts.BlockStage`, `ts.CausalLMHeadStage`, `ts.SimpleEmbedStage`): reusable building blocks for custom `extract` functions — no need to import private names.
- **`AuxInputStage`**: base class for stages that need extra named tensors (vision features, attention masks, …). Set `accepts_aux_inputs = True` and implement `forward(main, **aux) -> Tensor`. `SplitLayer` and all executors route aux tensors to the first stage only; all downstream stages receive a normal float activation.
- **`MoEBlockStage`**: wraps a block that returns `(hidden_states, scalar_aux_loss)` — MoE pattern (DeepSeek-V2/V3, Mixtral). Accumulates weighted aux loss during forward; executors call `pop_moe_aux_loss()` on the `SplitLayer` after each activation backward and run a local `.backward()` — router weights train correctly on each worker without crossing partition boundaries. DeepSeek-V2/V3 blocks are wrapped automatically by `wrap_hf()`.
- **Multi-modal DataLoader convention**: coordinator DataLoader may yield `({"input_ids": ..., "pixel_values": ...}, labels)` or `((main_tensor, aux_dict), labels)`. `_unpack_inputs()` in both executors handles all three forms; aux tensors are chunked per micro-batch when GPipe is enabled.
- **Proto: `map<string, Tensor> aux_inputs = 4`** on `ForwardRequest` — backward-compatible optional field; sent to worker 0 only; named keys survive the round-trip unchanged.
- **`SplitLayer.forward(input, **aux)`**: passes aux kwargs to the first sub-layer if it has `accepts_aux_inputs = True`; no-ops otherwise (zero overhead for existing runs). `SplitLayer.pop_moe_aux_loss()` sums and clears `MoEBlockStage` buffers across the partition.
- **LoRA + HF**: use `ts.peft_unwrap()` before `ts.wrap_hf()` to expose LoRA-injected weights. After `get_peft_model()`, explicitly `p.requires_grad_(True)` on `lm_head` — weight tying is broken at the serialization boundary, so the last worker's output projection must be unfrozen manually.
- **Worker train mode**: `torch.load()` on workers preserves the `eval()` mode set by `from_pretrained()`. `WorkerServicer.init()` calls `self.layer.train()` after `.to(device)` to ensure correct dropout/BN behavior during distributed training.

### Layer 9 — Monitoring (`torchslicer/monitor/`)
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
- **Per-batch loss logging**: `metrics.jsonl` receives one `phase="batch"` entry per batch (in addition to per-epoch entries) — enables `tail -f runs/<run_id>/metrics.jsonl` live monitoring; implemented in `LocalExecutor`, `DistributedExecutor`, and P2P training loop
- **Per-phase profiling**: `profile.verbosity` controls granularity (0=off, 1=epoch totals, 2=phase breakdown, 3=per-batch); workers instrument forward/backward/optimizer/send/idle phases; coordinator pulls via new `get_stats` RPC after each epoch
- **Callback system**: `SlicedModel.train(callbacks=[...])` accepts `TrainingCallback` subclasses; `on_epoch_end` can inject custom metrics (accuracy, lr, etc.) into `metrics.jsonl`
- **P2P logging & profiling**: `run_training()` accepts `run_logger`, `callbacks`, `follower_stubs`; writes `metrics.jsonl` (per-batch + per-epoch), `coordinator.jsonl`, `worker_epoch.jsonl`; driver stats pulled directly from profiler (no RPC), follower stats via `get_stats` RPC — same pattern as `DistributedExecutor`
- **GPipe in P2P**: epoch-boundary sync verified — `is_last_micro` logic in `P2PDriverServicer._send_backward` correctly fires `signal_batch_done()` only on the last micro-batch; `examples/test_p2p_gpipe.py` is an in-process integration test (synthetic data, n_micro=2, no Docker required)
- **int64 input support**: `WorkerServicer._backward()` handles `x_ref is None` (first partition with token-ID input) — `grad = x_ref.grad if x_ref is not None else None`; enables transformer / LLM experiments without any model restructuring
- **TinyGPT LM experiment**: `examples/train/lm/worker/main.py` — byte-level character LM, 4-block causal transformer (~830K params), P2P topology; `experiments/tinygpt_2gpu_p2p.yaml`; `make run-lm-gpu/run-lm-cpu` targets; verified: 10 epochs, 2 GPU workers, loss 3.03 → 0.75 on repo markdown corpus
- **LoRA/PEFT support**: `ts.peft_unwrap(peft_model)` extracts `model.base_model.model` so `ts.slice()` sees the original children after `get_peft_model()`; `LocalExecutor`, `WorkerServicer`, and P2P driver all filter optimizer params to `requires_grad=True` only (LoRA A/B matrices, not frozen base weights); `peft>=0.10` in Dockerfiles; workers can deserialise LoRA-modified layers via `torch.load()` without extra setup
- **TinyGPT+LoRA experiment**: `examples/train/lm/lora_worker/main.py` — phase 1: local full-param pre-training on driver (`PRETRAIN_EPOCHS`, default 5); phase 2: apply `LoraConfig(r=8, target_modules=["qkv"])`; phase 3: distributed LoRA fine-tuning (2 GPU workers, ~16K trainable/879K total); verified: pretrain loss 4.70→4.17, fine-tune loss 1.73→1.69; `make run-lora-gpu/run-lora-cpu` targets; `LORA_R`, `LORA_ALPHA`, `PRETRAIN_EPOCHS` env vars
- **HuggingFace adapter** (`torchslicer/adapters/hf.py`): `ts.wrap_hf(model, task)` wraps any HF model as `HFAdapter(nn.Sequential)` — flat sequence of opaque stage modules; no cross-partition skip connections; supports `causal_lm`, `seq_cls`, `masked_lm`; GPT-2/BERT/LLaMA/Qwen/DeepSeek architecture detection
- **Architecture registry** (`ts.register_hf_architecture`): plug in custom extractors (LLaVA, InternVL, Qwen-VL, …) without forking the library; checked before built-in detectors
- **`AuxInputStage`**: base class for multimodal stage 0; `SplitLayer.forward(**aux)` + proto `aux_inputs` map routes extra tensors to first worker only; DataLoader can yield dict inputs transparently
- **`MoEBlockStage`**: wraps MoE transformer blocks; accumulates router aux loss during forward; executors pop aux loss *before* calling `backward()` and add it to the primary loss (single backward pass through shared graph nodes); auto-detected by `wrap_hf()` for DeepSeek-V2/V3; fix applied in `LocalExecutor`, `_gpipe_batch`, and `WorkerServicer._run_backward_last`/`_backward`
- **RoPE fix (transformers ≥ 4.43)**: `_BlockStage` and `_MoEBlockStage` accept `rotary_emb`; `_stages_llama()` detects `core.rotary_emb` and passes it; stages compute `position_embeddings=(cos, sin)` per-forward — Mistral/Qwen/DeepSeek work with old and new transformers interfaces
- **`DistributedExecutor.reinit(model_graph, partitions, opt, crit)`**: re-sends `SliceConfig` to existing workers without restarting gRPC server or re-discovering; workers reset state via `init()` RPC; enables sequential multi-model runs on the same cluster
- **Qwen2/2.5 + DeepSeek-dense**: auto-detected via LLaMA branch (`model.model.embed_tokens` + `model.model.layers`); work with `wrap_hf()` unchanged
- **HF distributed experiments**: `examples/train/hf/coordinator/main.py` — `DistributedExecutor` with `CoordinatorDiscovery`; loads distilgpt2 + WikiText-2; supports baseline, LoRA, and GPipe modes from YAML; verified: 4 GPU workers, loss converges correctly in all three modes
- **HF experiment configs**: `experiments/hf_gpt2_4gpu_baseline.yaml` (b=16, 10 ep, lr=5e-5), `experiments/hf_gpt2_4gpu_lora.yaml` (r=32, α=64, c_attn+c_proj+c_fc, 15 ep, lr=3e-4), `experiments/hf_gpt2_4gpu_gpipe.yaml` (b=64, n_micro=4, 40 ep, lr=2e-4), `experiments/hf_gpt2_4gpu_baseline_b64.yaml` (b=64, 40 ep — GPipe fair comparison)
- **Worker train mode fix**: `WorkerServicer.init()` calls `self.layer.train()` after `torch.load()` + `.to(device)` — HF `from_pretrained()` leaves models in eval mode which suppresses dropout and prevents correct training; `LocalExecutor.setup()` similarly calls `sl.train()` on each split layer
- **lm_head unfreeze for LoRA split learning**: weight tying between `wte` and `lm_head` is broken at the serialization boundary when splitting; the last worker's `lm_head` must be explicitly unfrozen (`p.requires_grad_(True)`) after `get_peft_model()` and before `peft_unwrap()`, otherwise no gradient flows to the output projection
- **Device placement in LocalExecutor**: `inputs` and `labels` moved to device at start of `_standard_batch()` and `_gpipe_batch()` — required for GPU training where DataLoader returns CPU tensors
- **Epoch progress logging**: `train_epoch()` now accepts `total_epochs` and prints `[epoch N/M]` headers in both `LocalExecutor` and `DistributedExecutor`
- `COORDINATOR_ADDRESS` env var on workers; `WORKER_ADDRESS` for custom address advertisement
- `make run-cpu/run-gpu CONFIG=experiments/resnet18_4gpu.yaml` — one-command experiment launch
- `make run-p2p-gpu/run-p2p-cpu CONFIG=experiments/resnet18_2gpu_p2p.yaml` — P2P stack (no coordinator)
- `make run-lm-gpu/run-lm-cpu CONFIG=experiments/tinygpt_2gpu_p2p.yaml` — TinyGPT LM P2P stack
- `make run-lora-gpu/run-lora-cpu CONFIG=experiments/lora_tinygpt_2gpu_p2p.yaml` — TinyGPT+LoRA P2P stack
- `make run-hf-dist-gpu/run-hf-dist-cpu CONFIG=experiments/hf_gpt2_4gpu_baseline.yaml` — HF distributed stack (4 workers, centralized)
- `make run-hf-gpu/run-hf-cpu` — HF LocalExecutor fine-tuning (single container)
- `make run-arch-ext-gpu/run-arch-ext-cpu` — arch extension smoke test (2 workers, Mistral-tiny + DeepSeek-MoE-synthetic, verified)
- Docker stack verified: CPU (2–4 workers) and GPU (RTX 3060, WSL2 + NVIDIA Container Toolkit); P2P verified same

### What is incomplete
- **Fault tolerance**: crash aborts the run; `BaseDiscovery.watch(on_join, on_leave)` hook exists but nothing calls it; need a heartbeat thread per worker firing `on_leave` on missed pings; automatic re-slicing and coordinator failover are follow-ons
- **Multi-modal support (LLaVA etc.)**: `AuxInputStage` + proto `aux_inputs` field provide the plumbing; users must write and register the custom stage 0 (vision encoder + projector fusion). No built-in LLaVA/InternVL extractor yet.
- **HF fine-tuning granularity**: `HFAdapter` splits at transformer block boundaries. Intra-block splitting (attention vs FFN) requires cross-partition skip connections, which remain unsupported.
- **Dynamic cluster management**: slice assignment fixed at `init()` — no mid-run rebalancing; all workers get equal batch sizes regardless of device capability; `NodeInfo.memory_mb`/`device` are advertised but no splitter uses them yet; no intra-layer parallelism if a single layer exceeds one device's memory
- No REST transport logic (Dockerfiles ready)
- No `EnergySplitter` / `DeadlineSplitter`
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

### Run HuggingFace GPT-2 experiments (centralized, 4 workers)
```bash
# Baseline full fine-tuning
make run-hf-dist-gpu CONFIG=experiments/hf_gpt2_4gpu_baseline.yaml

# LoRA fine-tuning (r=32, c_attn + c_proj + c_fc)
make run-hf-dist-gpu CONFIG=experiments/hf_gpt2_4gpu_lora.yaml

# GPipe micro-batch pipelining (n_micro=4)
make run-hf-dist-gpu CONFIG=experiments/hf_gpt2_4gpu_gpipe.yaml

# LocalExecutor single-container (GPU)
make run-hf-gpu

# Tear down HF dist stack
make down-hf-dist
```

HF dist stack: `docker-compose.hf-dist.yml` + `docker-compose.hf-dist.gpu.yml` override.
HF local stack: `docker-compose.hf.yml` + `docker-compose.hf.gpu.yml` override.
Dataset downloads cached in `~/.cache/huggingface/` (bind-mounted as `HF_HOME`).

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
| `Dockerfile.cpu` | `python:3.12-slim` | torch-cpu + grpcio + fastapi + transformers + datasets + peft + torchslicer |
| `Dockerfile.gpu` | `python:3.12-slim` | torch-cu121 + grpcio + fastapi + transformers + datasets + peft + torchslicer |

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
