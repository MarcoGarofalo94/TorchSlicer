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
- [x] Full Docker stack verified: GPU stack runs coordinator + 2 workers (`device=cuda`), all 20 epochs complete cleanly (loss 2.16 → 0.56 on ResNet18/CIFAR-10)
- [x] Proto files live in `lib/torchslicer/torchslicer/transport/grpc/` (not in examples); auto-compiled at container startup

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
- [ ] Design P2P coordination protocol
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
- [ ] Device profiling (energy, latency)
- [ ] Gradient norm logging per slice
- [ ] Comparison harness: strategies × transports × topologies

## Optimization ✅ (partial)
- [x] Replace deprecated `Variable` with `detach().requires_grad_(True)` in `SplitLayer`
- [x] Raw-bytes tensor serialization — activation/gradient tensors now sent as contiguous buffer; `torch.frombuffer` on receive; `torch.save` kept only for layer/optimizer config (one-time startup)
- [x] Persistent gRPC stubs — `_next_stub`, `_prev_stub`, `_coord_stub` created at `init()`, reused across all batches
- [x] ThreadPoolExecutor at worker — pre-warmed pool (4 workers) replaces per-RPC thread spawn
- [x] Mixed-precision opt-in — `mixed_precision=True` on `SlicedModel.train()` / `executor.setup()`; wraps partition forward in `torch.autocast(bfloat16)`; no GradScaler needed (bfloat16 stable range)
- [x] GPU OOM fix for long runs — `SplitLayer.optimize()` nulls `self.x` after each batch; worker calls `torch.cuda.empty_cache()` after every backward; validated over 20 epochs without OOM
- [ ] Micro-batch pipeline parallelism (GPipe-style) — overlap next batch's forward with current batch's backward; `batch_id` keying already supports it
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
- Hostnames `worker1`, `worker2`, `coordinator` are hardcoded in the gRPC example
- `DistributedExecutor` workers receive layers as a flat sequential list; intra-partition DAG info is not sent over the wire (proto would need to carry predecessor indices). For standard models (ResNet etc.) the partitions are sequential anyway.
- `Transport`, `Topology`, `Monitor` in `lib/` are interfaces only; implementations live in `examples/`
