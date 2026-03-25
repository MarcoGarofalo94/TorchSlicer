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
- [x] Coordinator blocks on `signal.pause()` after training — exits only on `SIGTERM` (`make down`); workers stay up and accept next run without restart; `make _env` writes `UID`/`GID` to `.env` so run artifacts are owned by host user
- [x] Idle workers (registered but not selected by `discover()`) receive `Shutdown` RPC at `teardown()` via `BaseDiscovery.idle_nodes()` — all workers exit cleanly (code 0) regardless of selection
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
- [x] Makefile `CONFIG=` parameter — `make run-centralized CONFIG=experiments/resnet18_4gpu.yaml`
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
- [x] Intra-partition skip connections work end-to-end (local and distributed): `SliceConfig` carries `PredecessorList` per layer; worker passes to `SplitLayer`; verified with explicit skip-connection model over gRPC (`examples/test_dag_distributed.py`)
- [x] Cross-partition multi-input nodes raise `ValueError` with actionable hint
- [x] ResNet18/50 work natively — `torch.flatten` captured automatically, no wrapper needed
- [ ] Cross-partition skip connections (requires executor protocol changes)

## P2P topology ✅
- [x] `WorkerServicer` moved to library (`executors/worker.py`) — shared by centralized and P2P; centralized worker now imports from it
- [x] `P2PDriverServicer` — extends `WorkerServicer`; overrides `_send_backward` to signal batch completion directly (no loopback RPC); adds `run_own_forward()` for DataLoader-driven forward pass
- [x] `_P2PCoordinatorServicer` — lightweight embedded coordinator on driver; handles `report_metrics` from last worker and `signal_batch_done()` directly (no gRPC overhead on hot path)
- [x] Driver (worker 0) builds full model, runs splitter, sends `SliceConfig` to followers via `init()` RPC with retry; configures own partition locally (memory-efficient: each node holds only its slice)
- [x] Labels sent directly from driver to last worker — intermediate workers never see labels (split-learning privacy guarantee)
- [x] `IS_DRIVER` env var selects role; all workers run same script (`examples/train/p2p/worker/main.py`)
- [x] `WORKER_PEERS` env var (or `discovery.peers` in YAML) defines peer list; driver retries `init()` for 60s to handle startup race
- [x] Graceful shutdown: driver sends `Shutdown` RPC to all followers after training; followers exit cleanly (code 0)
- [x] `docker-compose.p2p.yml` + `docker-compose.p2p.gpu.yml`; `experiments/resnet18_2gpu_p2p.yaml`
- [x] Makefile `make run-p2p CONFIG=... DEVICE=cpu|gpu` targets
- [x] Smoke-tested: 20 epochs ResNet18/CIFAR-10, 2 GPU workers, both exit 0 (loss 2.4 → 0.52)
- [x] **GPipe in P2P** — epoch-boundary sync verified; `is_last_micro` gates `signal_batch_done()` correctly; `examples/test_p2p_gpipe.py` integration test passes (2 workers, n_micro=2, 2 epochs, in-process gRPC servers)
- [x] **Logging / profiling in P2P** — `RunLogger`, `WorkerProfiler`, and `get_stats` pull integrated into `examples/train/p2p/worker/main.py`; per-batch + per-epoch metrics, coordinator overhead, driver/follower profiler stats; `run_manifest.json` written at teardown; same pattern as `DistributedExecutor`
- [x] **int64 gradient fix** — `WorkerServicer._backward()` guards `x_ref.grad if x_ref is not None else None`; unblocks transformer/LLM experiments where first partition input is token IDs (non-float → `SplitLayer` already sets `self.x = None`)
- [x] **TinyGPT LM experiment** — `examples/train/lm/worker/main.py`; byte-level character LM (vocab=256, d_model=128, n_heads=4, 4 transformer blocks, ~830K params); corpus from `/workspace/*.md`; `experiments/tinygpt_2gpu_p2p.yaml`; launch via `make run-p2p DEVICE=gpu CONFIG=experiments/tinygpt_2gpu_p2p.yaml`
- [x] **3-worker P2P** — `experiments/resnet18_3worker_p2p.yaml`; `make run-p2p DEVICE=gpu CONFIG=experiments/resnet18_3worker_p2p.yaml`; driver sends slices to 2 followers; N-worker P2P handled generically via `n = len(peers)`
- [ ] `MDNSDiscovery` for zero-config local network (future)

## LoRA / PEFT support ✅
- [x] **`ts.peft_unwrap(model)`** — unwraps `PeftModel` (from `get_peft_model()`) to expose its inner model for `ts.slice()`; returns `model.base_model.model` with LoRA adapters injected in-place; idempotent on non-PEFT models; raises helpful `ImportError` if `peft` not installed
- [x] **Trainable-params filter** — `LocalExecutor`, `WorkerServicer.init()`, and both P2P driver `_configure_driver_slice()` functions now build the optimizer over `[p for p in params if p.requires_grad]`; frozen base weights (LoRA pattern) are excluded automatically, avoiding wasted optimizer state; falls back to all params if none are trainable
- [x] **`peft` in Dockerfiles** — added to `Dockerfile.cpu` and `Dockerfile.gpu` Layer 2; workers can `torch.load()` LoRA-modified layers without extra setup
- [x] **`peft>=0.10` optional dep** in `pyproject.toml` (`pip install torchslicer[peft]`); not required for non-PEFT usage
- [x] **TinyGPT + LoRA example** (`examples/train/lm/lora_worker/main.py`) — same P2P topology as LM example; applies `LoraConfig(r=8, target_modules=["qkv"])` before slicing; logs trainable vs total param count; `LORA_R` / `LORA_ALPHA` env vars for override
- [x] `experiments/lora_tinygpt_2gpu_p2p.yaml` — AdamW lr=3e-4, 10 epochs, 2-worker P2P
- [x] `experiments/lora_tinygpt_2gpu_p2p.yaml`; launch via `make run-p2p DEVICE=gpu CONFIG=experiments/lora_tinygpt_2gpu_p2p.yaml`

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
- [x] **`RunLogger`** (`monitor/run_logger.py`) — per-run artifact writer/reader; writes `run_manifest.json` + per-phase JSONL files to `{logging.dir}/{run_id}/`; each file is schema-homogeneous (`pd.read_json(path, lines=True)` just works); `load(run_dir)` + `to_dataframe(phase)` for plotting; checkpoints co-located when enabled
- [x] **Per-batch logging** — `phase="batch"` maps to `metrics.jsonl` in `RunLogger`; all executors (`LocalExecutor`, `DistributedExecutor`, P2P driver) call `run_logger.log()` inside the batch loop; file updates are immediate (unbuffered), enabling `tail -f runs/{run_id}/metrics.jsonl` as a live training monitor
- [x] **Per-phase log files**: `metrics.jsonl` (epoch + per-batch loss/duration), `coordinator.jsonl` (overhead), `worker_epoch.jsonl` (per-worker aggregates with avg/min/max/p95 + GPU memory), `worker_batch.jsonl` (verbosity=3, per-batch detail), `partition_epoch/batch.jsonl` (local executor)
- [x] **`TrainingCallback`** (`monitor/callback.py`) — base class for training hooks; `on_epoch_end(epoch, metrics) -> dict` lets users inject custom metrics (accuracy, lr, etc.) into `metrics.jsonl`; passed via `SlicedModel.train(callbacks=[...])`
- [x] **`WorkerProfiler`** (`monitor/profiler.py`) — worker-side per-phase timer (forward, backward, optimizer, send_fwd, send_bwd, idle_fwd, idle_bwd); verbosity=0 is zero-overhead; optional GPU memory snapshots; coordinator pulls via `get_stats` RPC after each epoch
- [x] **`get_stats` RPC** on `worker_service.proto` — `GetStatsRequest` / `WorkerStatsResponse` with `PhaseStats` (avg/min/max/p95/total) per phase; `BatchStats` list at verbosity=3; resets worker profiler after each pull
- [x] **Coordinator-side overhead logging** — `data_load_total_ms`, `send_total_ms`, `wait_total_ms` per epoch logged to `coordinator.jsonl`
- [x] **Unified run directory** — logs + checkpoints in same `{logging.dir}/{run_id}/`; files owned by host user (UID/GID injected via `.env` by `make _env`)
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

## Fault tolerance ✅ (centralized topology)
- [x] **`FaultToleranceConfig`** — `enabled`, `heartbeat_interval_s`, `ping_timeout_s`, `max_retries`; env vars `FAULT_TOLERANCE_ENABLED`, `HEARTBEAT_INTERVAL`, `HEARTBEAT_TIMEOUT`, `FAULT_MAX_RETRIES`; YAML key `fault_tolerance`; added to `RunConfig`
- [x] **Heartbeat thread per worker** — `DistributedExecutor._heartbeat_loop()` pings `get_stats(epoch=-1, verbosity=0)` with short timeout; `_on_worker_failure()` sets `_failure_event` and unblocks `_batch_done.wait()`; one thread per worker, started at `setup()`
- [x] **`save_checkpoint` RPC** — new RPC on `WorkerService` (reuses `ShutdownRequest`/`Ack`); saves slice state without stopping the server; `WorkerServicer.save_checkpoint()` calls `_save_checkpoint()`; coordinator calls it on all workers after every epoch when FT enabled
- [x] **`WorkerFailureError`** — raised when a gRPC error occurs mid-batch and `_failure_event` is set; caught in `SlicedModel.train()` retry loop; `worker_idx`, `address`, `epoch` attributes
- [x] **`DistributedExecutor.recover(last_good_epoch)`** — stops heartbeats → identifies survivors → loads epoch-N checkpoint state_dicts into `_stored_layers` in-place → uniform re-partition → reinit survivors via `init()` RPC → resets failure state → restarts heartbeats
- [x] **Retry loop in `SlicedModel.train()`** — catches `WorkerFailureError`, calls `executor.recover(epoch - 1)`, retries the failed epoch; raises after `max_retries` attempts
- [x] **`FaultTestWorkerServicer`** (`examples/train/centralized/GRPC/worker/main_ft_test.py`) — crash injection via `os._exit(137)` 2s after target epoch's `save_checkpoint` RPC; `FAULT_KILL_EPOCH` env var
- [x] **`experiments/resnet18_3worker_ft.yaml`** — 3-worker FT test, 6 epochs, `fault_tolerance.enabled=true`, `heartbeat_interval_s=3.0`, `ping_timeout_s=2.0`, `max_retries=2`
- [x] **Fault injection test** — `experiments/resnet18_3worker_ft.yaml` + `examples/train/centralized/GRPC/worker/main_ft_test.py` (`FaultTestWorkerServicer`); `worker2` crashes after epoch 2; run via `make run-centralized` with `COORDINATOR_CMD` and the ft config
- [x] **Verified** — worker2 crashes mid epoch-3; heartbeat detects failure; `recover(2)` re-slices to 2 survivors; epochs 3–6 continue; loss 3.09→2.75→2.49→2.26→2.06→1.98; clean coordinator exit
- [x] **entrypoint.sh fix** — `LOGNAME=${LOGNAME:-user}` + `HOME=${HOME:-/tmp}` exported before exec; fixes `getpass.getuser()` → `pwd.getpwuid()` crash when container runs as numeric UID not in `/etc/passwd` (affects all compose files using `user: "${UID}:${GID}"`)
- [ ] **Coordinator failover** — if the coordinator process crashes, all workers stall; `run_id` on all messages is the hook for re-registration on restart; not implemented

## HuggingFace / transformer support ✅
- [x] **`ts.peft_unwrap(model)` + `pack=` hook** — primary path for HF models: `ts.slice(ts.peft_unwrap(model), n=4, pack=pack_gpt2)`; architecture-specific `pack` functions in `examples/pack/`
- [x] **`CausalLMHeadStage`** — transposes output to `[B, V, T]` for `CrossEntropyLoss` compatibility (HF lm_head outputs `[B, T, V]`)
- [x] **Worker train mode fix** — `WorkerServicer.init()` calls `self.layer.train()` after `torch.load()` + `.to(device)`; `LocalExecutor.setup()` calls `sl.train()` on each partition; fixes silent eval-mode dropout suppression when using `from_pretrained()` weights
- [x] **LoRA split learning: lm_head unfreeze** — weight tying between `wte` and `lm_head` is broken at serialization; last worker must train its own output projection; fix: `p.requires_grad_(True)` on `lm_head` before `peft_unwrap()` in coordinator
- [x] **Device placement fix** — `LocalExecutor._standard_batch()` and `_gpipe_batch()` move `inputs`/`labels` to device at batch start; required for GPU training with CPU DataLoader tensors
- [x] **Epoch progress logging** — `train_epoch()` accepts `total_epochs`; both `LocalExecutor` and `DistributedExecutor` print `[epoch N/M]` headers
- [x] **`transformers>=4.40.0` + `datasets>=2.18.0`** added to `Dockerfile.cpu` and `Dockerfile.gpu`; `HF_HOME=/workspace/.cache/huggingface` bind-mount in compose coordinator for dataset/model cache persistence
- [x] **HF distributed experiment** (`examples/train/hf/coordinator/main.py`) — `DistributedExecutor` + `CoordinatorDiscovery`; loads distilgpt2 + WikiText-2 via `pack_gpt2`; three modes (baseline, LoRA, GPipe) from YAML; verified 4-GPU workers; launch via `make run-centralized DEVICE=gpu CONFIG=... COORDINATOR_CMD="python3 examples/train/hf/coordinator/main.py"`
- [x] **Experiment configs**: `hf_gpt2_4gpu_baseline.yaml` (b=16, 10 ep, lr=5e-5), `hf_gpt2_4gpu_lora.yaml` (r=32, α=64, c_attn+c_proj+c_fc, 15 ep, lr=3e-4), `hf_gpt2_4gpu_gpipe.yaml` (b=64, n_micro=4, 40 ep, lr=2e-4)
- [x] **Verified results**: baseline loss 3.86→1.67 (10 ep); LoRA loss 3.95→1.44 (15 ep, 3.26% trainable); GPipe throughput ~1.57× vs sequential at same gradient-step budget
- [ ] **Intra-block splitting** — splitting attention from FFN within a block would require cross-partition skip connections; not supported; current granularity is one block per stage

## Stage primitives & architecture support ✅
- [x] **`torchslicer/core/stages.py`** — `GPT2EmbedStage`, `SimpleEmbedStage`, `BlockStage`, `CausalLMHeadStage`, `AuxInputStage`, `MoEBlockStage`; all exported at `ts.*`; workers can always deserialize without extra imports
- [x] **`AuxInputStage`** — base class (`accepts_aux_inputs=True`) for multimodal stage 0; `forward(main, **aux) -> Tensor` contract; `SplitLayer.forward(input, **aux)` passes kwargs to first sub-layer only if it declares `accepts_aux_inputs`; zero overhead for all other stages
- [x] **`MoEBlockStage`** — wraps MoE blocks returning `(hidden_states, scalar_aux_loss)`; accumulates weighted aux loss during forward; `pop_aux_loss()` clears buffer; `SplitLayer.pop_moe_aux_loss()` sums across partition; executors call it after activation backward and run a local `.backward()` — router weights trained correctly without crossing partition boundaries
- [x] **Proto: `map<string, Tensor> aux_inputs = 4`** on `ForwardRequest` — backward-compatible optional field; coordinator serializes aux tensors (pixel_values etc.) and sends to worker 0 only; workers deserialize and unpack into `**aux` kwargs; empty map = zero overhead
- [x] **Multi-modal DataLoader convention** — coordinator accepts `({"input_ids":…, "pixel_values":…}, labels)`, `((main, aux_dict), labels)`, or plain `(Tensor, labels)`; `_unpack_inputs()` helper in both `LocalExecutor` and `DistributedExecutor`; aux tensors chunked per micro-batch when GPipe is active
- [x] **MoE aux loss graph fix** — pop aux loss *before* calling `backward()` and combine with primary loss (`total = loss + moe`) to avoid double-backward on shared graph nodes; non-last partitions use `retain_graph=True`; fix applied in `LocalExecutor`, `DistributedExecutor._gpipe_batch`, and `WorkerServicer`
- [x] **`DistributedExecutor.reinit()`** — re-sends `SliceConfig` to existing workers without restarting the gRPC server or re-discovering workers; enables sequential multi-model tests on the same cluster
- [x] **Arch extension smoke test** — `examples/test_arch_extension.py`: Mistral-tiny (from config, no download) + synthetic DeepSeek-V2 MoE; both pass with `LocalExecutor` (2 partitions, 2 epochs, CPU/GPU)
- [x] **Arch extension distributed test** — `examples/test_arch_ext_dist/coordinator/main.py` + custom worker; runs both models sequentially on same 2 GPU workers via `reinit()`; verified: Mistral 6.26→6.04, DeepSeek-MoE 6.39→6.23, both workers exit 0
- [ ] **Built-in LLaVA/InternVL/Qwen-VL pack functions** — `AuxInputStage` plumbing is ready; users must write a custom `pack` function with a multimodal stage 0

## Dynamic cluster management
- [ ] **Dynamic re-slicing** — no way to add/remove workers mid-run or rebalance partitions; the slice assignment is fixed at `init()` time for the entire training run
- [ ] **Heterogeneous batch routing** — all workers receive the same micro-batch size; devices with more memory (or faster compute) could be assigned larger shares; `NodeInfo.memory_mb` and `NodeInfo.device` are already advertised at registration but not used by any splitter
- [ ] **Intra-layer parallelism** — if a single layer is too large for one device's memory, there is no way to split it further; the current model requires every direct child of the root to fit on one worker

## Platform support
- [ ] ARM64 (aarch64) support — Dockerfiles use x86-only PyTorch wheel URLs; need ARM64-compatible base image and wheel source

## Library API & packaging refactor ✅

### 1. `pack=` hook on `ts.slice()`
- [x] Add `pack: Callable[[nn.Module], list[nn.Module]] | None = None` parameter to `ts.slice()`
- [x] Default: existing torch.fx shallow trace path (works for ResNet, VGG, sequential models)
- [x] When provided: skip tracing entirely; build `ModelGraph` via `ModelGraph.from_stages(stages)`
- [x] `ModelGraph.from_stages(stages)`: builds a sequential DAG from a flat stage list, no fx tracing

### 2. Strip HF-specific code from library → moved to examples
- [x] Removed `HFAdapter`, `wrap_hf()`, `register_hf_architecture()`, all `_stages_*` detectors from library
- [x] Stage primitives (`BlockStage`, `GPT2EmbedStage`, `AuxInputStage`, `MoEBlockStage`, `CausalLMHeadStage`, `SimpleEmbedStage`) moved to `torchslicer/core/stages.py`; all exported at `ts.*`
- [x] `torchslicer/__init__.py` updated; `torchslicer/adapters/hf.py` emptied
- [x] Reference pack functions in `examples/pack/`: `gpt2.py`, `llama.py`, `bert.py`, `deepseek_moe.py`; each file self-contained and copy-pasteable; `__init__.py` documents the serialization constraint

### 3. GPU/CPU device handling
- [x] `resolve_device()` in `executors/worker.py` — `DEVICE=auto` uses CUDA if available, else CPU
- [x] `IMAGE_TAG=cpu|gpu` selects image in compose; `DEVICE=gpu` on make targets sets both `IMAGE_TAG=gpu` and `DEVICE=cuda`
- [x] Thin GPU override files (`docker-compose.gpu.yml`, `docker-compose.p2p.gpu.yml`) add only NVIDIA device reservations — no feature duplication

### 4. Compose files collapsed
- [x] `docker-compose.yml` — centralized (coordinator + N workers); `COORDINATOR_CMD` overrides coordinator script
- [x] `docker-compose.p2p.yml` — P2P (N workers, `IS_DRIVER` selects role)
- [x] `docker-compose.gpu.yml` + `docker-compose.p2p.gpu.yml` — thin GPU overrides (NVIDIA reservations only)
- [x] All feature-specific compose files removed (lm, lora, hf, hf-dist, ft-test, arch-ext, p2p3 and all their .gpu.yml variants)
- [x] Makefile: `make run-centralized CONFIG=...` and `make run-p2p CONFIG=...`, each with `DEVICE=cpu|gpu` toggle; `COORDINATOR_CMD=` for custom coordinator script; `HF_HOME`/`HF_TOKEN` forwarded to coordinator

### 5. Reference pack functions
- [x] `examples/pack/gpt2.py` — `pack_gpt2` for GPT-2 / DistilGPT-2 (uses `ts.GPT2EmbedStage`, `ts.BlockStage`, `ts.CausalLMHeadStage`)
- [x] `examples/pack/llama.py` — `pack_llama` for LLaMA 2/3, Mistral, Qwen2/2.5, DeepSeek-dense
- [x] `examples/pack/bert.py` — `pack_bert_seq_cls`, `pack_bert_mlm` (uses `nn.Sequential` + `ts.BlockStage`, no custom classes)
- [x] `examples/pack/deepseek_moe.py` — `pack_moe` for DeepSeek-V2/V3 MoE (uses `ts.MoEBlockStage`)
- [x] `examples/pack/__init__.py` documents serialization constraint + three options for custom stage classes

## External comparisons
- [ ] Compare against AIRllm
- [ ] Compare against FedAvg / standard DDP
- [ ] Benchmark on hybrid CPU/GPU clusters

---

## Known limitations
- Cross-partition skip connections not supported — if a multi-input node's predecessors span partition boundaries, `validate()` raises `ValueError`. Workaround: choose `n` so skip connections stay within one partition, or wrap the block in a single `nn.Module` (which is what `pack` functions do for HF models).
- Worker hostnames no longer hardcoded — `CoordinatorDiscovery` replaces the `worker{i}` loop; workers advertise their own address at registration time
- `Transport`, `Topology` in `lib/` are interfaces only; implementations live in `examples/`
- Worker fault tolerance implemented for centralized topology — heartbeat detection, checkpoint-based recovery, re-slicing across survivors; coordinator failover not yet implemented
- HuggingFace models are supported via `pack=` hook at transformer block granularity; intra-block splitting requires cross-partition skip connections (unsupported). Multi-modal models need a custom `pack` function with an `AuxInputStage` as stage 0.
- All workers receive equal-sized batches regardless of device capability — `NodeInfo` carries device/memory info but no splitter or executor uses it for load balancing yet
- **`pack=` serialization constraint**: every class instantiated inside a `pack` function must be importable on workers under the same module path. Library classes (`BlockStage`, `GPT2EmbedStage`, etc.) are always safe. Custom classes require packaging or `PYTHONPATH` setup on workers.
