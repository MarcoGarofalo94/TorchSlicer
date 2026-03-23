# TorchSlicer — Progress

## Refactor & clean API ✅
- [x] Define `BaseSplitter`, `BaseExecutor`, `BaseTransport`, `BaseTopology` interfaces
- [x] Restructure `lib/torchslicer` into layered architecture (`core/`, `strategies/`, `executors/`, `transport/`, `topology/`, `monitor/`)
- [x] `Slicer` refactored — replaced inspect/regex/eval metadata mining with `torch.save(layer)` per layer; works for any `nn.Module`, architecture + weights in one shot
- [x] `SplitLayer` — forward, backward, optimize fully working
- [x] Top-level `ts.slice()` API + `SlicedModel.train()`
- [x] `pyproject.toml` + updated `setup.py` (find_packages)
- [x] Conda env `torchslicer` (Python 3.12)
- [x] Docker images consolidated: 11 Dockerfiles → `Dockerfile.cpu` + `Dockerfile.gpu` + `entrypoint.sh`
- [x] `entrypoint.sh` auto-recompiles `.proto` files at container startup (eliminates gencode/runtime version mismatch)
- [x] `docker-compose.gpu.yml` override — workers get NVIDIA device reservations; tested on RTX 3060
- [x] `docker-compose.monitor.yml` overlay — Jaeger + dashboard service; compose with base or GPU stack
- [x] Makefile simplified to `build-cpu`, `build-gpu`, `push`, `clean`
- [x] gRPC proto redesigned: typed messages (`SliceConfig`, `LayerConfig`, `ForwardRequest`, `BackwardRequest`), `batch_id` on every message, label travels in `ForwardRequest` (eliminates `set_label` race)
- [x] Coordinator and worker rewritten to match new proto and Slicer API; `batch_id`-keyed label/output dicts with proper locking

## Local executor ✅
- [x] `LocalExecutor` — single-device sequential training, no networking
- [x] `verbose=True` flag on `train_epoch` / `SlicedModel.train()` for per-batch loss logging
- [x] Smoke test: `examples/test_local_dnn.py` (synthetic data, 4 partitions, 5 epochs)
- [x] Fine-tune example: `examples/finetune/bert_sst2.py` (BERT SST-2)

## Distributed executor ✅
- [x] `DistributedExecutor` — centralized topology; this process acts as coordinator
- [x] Embedded gRPC coordinator server (`_CoordinatorServicer`) with `threading.Event` synchronisation
- [x] Sends `SliceConfig` to each worker at setup (layers serialized via `torch.save`, optimizer, criterion)
- [x] Synchronous training loop: sends batch → waits for `batch_done` callback → accumulates loss
- [x] Full Docker stack verified: GPU stack runs coordinator + 2–4 workers (`device=cuda`), all 20 epochs complete cleanly (loss 2.16 → 0.56 on ResNet18/CIFAR-10)
- [x] `N_WORKERS` env var — coordinator dynamically builds worker list; `docker-compose.yml` ships with worker1–worker4; scale up without rebuilding images
- [x] `EPOCHS`, `USE_GPIPE`, `N_MICRO`, `N_WORKERS` all forwarded into coordinator container via `environment:` in compose
- [x] Proto files live in `lib/torchslicer/torchslicer/transport/grpc/` (not in examples); auto-compiled at container startup

## Discovery & lifecycle ✅
- [x] `BaseDiscovery` abstraction — `announce()`, `discover()`, `watch()` (fault-tolerance hook) in `torchslicer/discovery/base.py`
- [x] `CoordinatorDiscovery` — workers push `Register` RPC to coordinator at startup; coordinator waits for `N_WORKERS` registrations before dispatching `SliceConfig`; hardcoded `worker{i}` hostname generation eliminated
- [x] `StaticDiscovery` — peer list from config; no central point; P2P path for Docker and fixed-address clusters
- [x] `announce_to_coordinator()` — client-side helper used by workers at startup; retries until coordinator is reachable
- [x] `NodeInfo` proto message — node_id, address, device, memory_mb; carried in `RegisterRequest` / `RegisterResponse`; `run_id` assigned by coordinator, echoed in `batch_done` / `report_metrics` / `SliceConfig`
- [x] `Shutdown` RPC on `WorkerService` — graceful stop; `save_checkpoint` flag triggers slice save before `server.stop()`; runs in background thread so RPC returns `Ack` before shutdown
- [x] `DistributedExecutor.teardown()` — calls `Shutdown` on all workers before stopping own gRPC server; all containers exit cleanly
- [x] Worker state reset on `init()` — `_reset_state()` clears all per-batch dicts/stubs; workers reusable across runs without container restart
- [x] `--abort-on-container-exit` in Makefile `run-*` targets — compose tears down all workers when coordinator exits
- [x] Optional checkpoint — each worker saves `{checkpoint_dir}/{run_id}/worker_{index}_epoch_{n}.pt` with layer + optimizer state; coordinator saves `run_state.json`; resume via `checkpoint_path` field in `SliceConfig`; disabled by default (`CHECKPOINT_ENABLED=0`)
- [x] `run_id` on all proto messages — forward-compat hook for fault tolerance (coordinator restart detection, worker re-registration)
- [x] `COORDINATOR_ADDRESS` env var on workers — no more hardcoded coordinator hostname; defaults to `coordinator:50054`
- [x] `WORKER_ADDRESS` env var — lets workers advertise a custom address to the coordinator (useful behind NAT or custom Docker networks)

## Configuration system ✅
- [x] `RunConfig` dataclass with `TrainingConfig`, `PipelineConfig`, `DiscoveryConfig`, `CheckpointConfig`, `LoggingConfig`, `ProfileConfig` sub-dataclasses
- [x] `RunConfig.from_yaml(path)` — load from YAML experiment config file (requires PyYAML)
- [x] `RunConfig.from_env()` — load from environment variables
- [x] `RunConfig.load(path)` — merged load: YAML base + env var overrides; path defaults to `EXPERIMENT_CONFIG` env var
- [x] `--config` CLI flag on `coordinator/main.py` — pass YAML path at runtime
- [x] `.env.example` — documents all env vars with defaults; Docker Compose auto-reads `.env`
- [x] `EXPERIMENT_CONFIG` env var — path to YAML; when set, YAML values used as base with env var overrides on top
- [x] `experiments/resnet18_4gpu.yaml` + `experiments/resnet18_2cpu.yaml` — ready-to-use experiment configs
- [x] Makefile `CONFIG=` parameter — `make run-gpu CONFIG=experiments/resnet18_4gpu.yaml`
- [x] `LoggingConfig(enabled, dir)` — controls run artifact output; env vars `LOG_ENABLED`, `LOG_DIR`
- [x] `ProfileConfig(verbosity, memory)` — controls worker profiling granularity; env vars `PROFILE_VERBOSITY`, `PROFILE_MEMORY`

## Splitting strategies (partial) ✅
- [x] `BaseSplitter` abstract class (public API for user-defined strategies)
- [x] `UniformSplitter` — `math.ceil(n_layers / n_partitions)` per partition
- [x] `UniformSplitter` populates intra-partition DAG predecessor info from `ModelGraph`
- [x] Strategy registry — `ts.slice(model, strategy="uniform")` or by instance/class
- [x] `BaseSplitter.validate()` — coverage check + cross-partition multi-input node detection
- [ ] `EnergySplitter`
- [ ] `DeadlineSplitter`

## ModelGraph DAG ✅
- [x] `ModelGraph.from_module()` — traces with `torch.fx` using a shallow tracer (direct children are leaves)
- [x] Functional ops auto-wrapped: `torch.flatten` → `_FlattenWrapper`, `operator.add` → `_AddWrapper`
- [x] `ModelGraph.is_dag()` — detects non-sequential graphs
- [x] `Partition.predecessors` — intra-partition DAG edges carried from splitter to `SplitLayer`
- [x] `SplitLayer` DAG-aware `forward()` — per-layer output caching, multi-input routing
- [x] Intra-partition skip connections work (e.g. skip model with n=1)
- [x] Cross-partition multi-input nodes raise `ValueError` with actionable hint
- [x] ResNet18/50 work natively — `torch.flatten` captured automatically, no wrapper needed
- [ ] Cross-partition skip connections (requires executor protocol changes)

## P2P topology
- [ ] Design P2P coordination protocol — `StaticDiscovery` is the first building block; `MDNSDiscovery` for zero-config local network
- [ ] Implement `P2PTopology`

## REST transport
- [ ] Implement `RESTTransport` matching `GRPCTransport` interface (Dockerfiles already ready)

## Monitoring & benchmarking ✅ (partial)
- [x] OpenTelemetry tracer (`monitor/tracer.py`) — `configure()`, `span()` context manager, auto-configures from env, safe no-op if OTEL unavailable
- [x] `LocalExecutor` and `DistributedExecutor` emit OTel spans (batch, forward, backward) with batch_id, layer names, timing, memory usage
- [x] `docker-compose.monitor.yml` overlay — adds Jaeger (port 16686) + dashboard service (port 8080), sets OTEL env vars
- [x] Dashboard backend (`examples/monitor/app.py`) — FastAPI + WebSocket; polls Jaeger, accumulates batch/topology state server-side
- [x] Dashboard frontend (`examples/monitor/frontend/`) — React 18 + Recharts + Vite; topology panel, training loss chart, timeline swimlane, batches table
- [x] Topology visualization — per-worker card with layer names, parameter count, GPU memory bars
- [x] Timeline tab — swimlane chart with per-worker forward/backward timing per batch
- [x] Dashboard auto-resets on new run — detects new `worker.init` span timestamps, clears stale batch data so frontend never shows data from a previous run
- [x] Dashboard Node build OOM fix — `NODE_OPTIONS=--max-old-space-size=512` caps V8 heap during Vite build; `restart: on-failure` in compose overlay
- [x] **`RunLogger`** (`monitor/run_logger.py`) — per-run artifact writer/reader; writes `{logging.dir}/{run_id}/run_manifest.json` + `metrics.jsonl`; `load(run_dir)` + `to_dataframe()` for pandas-based plotting; checkpoints co-located when enabled
- [x] **`TrainingCallback`** (`monitor/callback.py`) — base class for training hooks; `on_epoch_end(epoch, metrics) -> dict` lets users inject custom metrics (accuracy, lr, etc.) into `metrics.jsonl`; passed via `SlicedModel.train(callbacks=[...])`
- [x] **`WorkerProfiler`** (`monitor/profiler.py`) — worker-side per-phase timer (forward, backward, optimizer, send_fwd, send_bwd, idle_fwd, idle_bwd); verbosity=0 is zero-overhead; optional GPU memory snapshots; coordinator pulls via `get_stats` RPC after each epoch
- [x] **`get_stats` RPC** on `worker_service.proto` — `GetStatsRequest` / `WorkerStatsResponse` with `PhaseStats` (avg/min/max/p95/total) per phase; `BatchStats` list at verbosity=3; resets worker profiler after each pull
- [x] **Coordinator-side overhead logging** — `data_load_total_ms`, `send_total_ms`, `wait_total_ms` per epoch logged as `"coordinator_epoch"` phase in `metrics.jsonl`
- [x] **Unified run directory** — logs + checkpoints in same `{logging.dir}/{run_id}/`; `shutdown` sends the unified path so worker `.pt` files land alongside `run_manifest.json`
- [ ] Device profiling (energy, latency)
- [ ] Gradient norm logging per slice
- [ ] Comparison harness: strategies × transports × topologies

## Optimization ✅ (partial)
- [x] Replace deprecated `Variable` with `detach().requires_grad_(True)` in `SplitLayer`
- [x] Raw-bytes tensor serialization — activation/gradient tensors now sent as contiguous buffer; `torch.frombuffer` on receive; `torch.save` kept only for layer/optimizer config (one-time startup)
- [x] Persistent gRPC stubs — `_next_stub`, `_prev_stub`, `_coord_stub` created at `init()`, reused across all batches
- [x] ThreadPoolExecutor at worker — pre-warmed pool (max_workers=1 for compute serialization) replaces per-RPC thread spawn; gRPC server retains 10-worker pool for RPC handling
- [x] Mixed-precision opt-in — `mixed_precision=True` on `SlicedModel.train()` / `executor.setup()`; wraps partition forward in `torch.autocast(bfloat16)`; no GradScaler needed (bfloat16 stable range)
- [x] GPU OOM fix for long runs — `SplitLayer.optimize()` nulls `self.x` after each batch; worker calls `torch.cuda.empty_cache()` after every backward; validated over 20 epochs without OOM
- [x] GPipe micro-batch pipeline parallelism — opt-in via `use_gpipe=True, n_micro_batches=4` on `SlicedModel.train()`; `USE_GPIPE=1 N_MICRO=4` env vars for Docker; each micro-batch loss scaled by 1/M for correct gradient accumulation; single optimizer step after all M micro-batches; **benchmarked: 1.9× speedup with 4 workers** (282s → 149s / 5 epochs, ResNet18/CIFAR-10 GPU); 2-worker case is slower due to gRPC overhead exceeding pipeline gain; compute pool serialized (max_workers=1) to prevent concurrent backward races on shared model weights; x_ref saved per batch_id to avoid cut-point tensor overwrite across micro-batches
- [ ] More efficient gradient propagation beyond basic autograd

## Platform support
- [ ] ARM64 (aarch64) support — Dockerfiles use x86-only PyTorch wheel URLs; need ARM64-compatible base image and wheel source

## External comparisons
- [ ] Compare against AIRllm
- [ ] Compare against FedAvg / standard DDP
- [ ] Benchmark on hybrid CPU/GPU clusters

---

## Known limitations
- Cross-partition skip connections not supported — if a multi-input node's predecessors span partition boundaries, `validate()` raises `ValueError`. Workaround: choose `n` so skip connections stay within one partition, or wrap the block in a single `nn.Module`.
- Worker hostnames no longer hardcoded — `CoordinatorDiscovery` replaces the `worker{i}` loop; workers advertise their own address at registration time
- `DistributedExecutor` workers receive layers as a flat sequential list; intra-partition DAG info is not sent over the wire (proto would need to carry predecessor indices). For standard models (ResNet etc.) the partitions are sequential anyway.
- `Transport`, `Topology`, `Monitor` in `lib/` are interfaces only; implementations live in `examples/`
- No fault tolerance — coordinator or worker crash aborts the run; checkpoint support mitigates data loss; heartbeat-based failure detection is a future addition via `BaseDiscovery.watch()` (hook already in place)
