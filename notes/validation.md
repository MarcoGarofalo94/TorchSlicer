# Correctness validation: split training == unsplit training

Record for the TorchSlicer paper: what we checked, what broke, why, and how it was fixed.
Append new rounds at the end; don't rewrite past results.

## Claim

For a fixed seed, training a model sliced by TorchSlicer produces the same loss trajectory
as training the unsplit model in plain PyTorch. This holds in every execution mode (local,
GPipe, centralized, P2P), with both tensor transports (gRPC, raw TCP), on CPU and GPU, and
for both CNNs and Hugging Face LLMs, LoRA included. The one exception is GPipe with
BatchNorm, explained under "Known, by-design difference" below.

## Method (round 1, 2026-10-15, branch `feat/m0-foundations`)

- **Harness:** `benchmarks/equivalence/`. Run `run_matrix.py` for all models, or
  `run_matrix.py <model>` for one model, and see `tsmodels.py` for the setup. Every mode runs
  in separate OS processes, on 127.0.0.1 with real gRPC/TCP sockets.
- **Reference:** the unsplit model trained in plain PyTorch, with the same optimizer, the
  same data order and the same initial weights (`torch.manual_seed(0)` before construction,
  and an unshuffled DataLoader built from a generator seeded with 1). For the HF models the
  reference chains the `pack` stages, and `run_inproc.py` asserts that this forward equals
  the native HF forward to within 1e-5 before training.
- **Models (9):**
  - DNN (MLP), a small CNN, ResNet-18 (CIFAR stem) and TinyGPT (`examples/train_lm_local.py`).
  - GPT-2 and Llama, built from small HF configs (4 layers, d=64, vocab 100) and split with
    `examples/pack/{gpt2,llama}.py`.
  - LoRA on TinyGPT, GPT-2 and Llama (peft, r=4, frozen base).
- **Modes (6):**
  - `local` and `gpipe`, both `LocalExecutor`, GPipe with 2 micro-batches.
  - `central-grpc` and `central-tcp`: a coordinator plus 3 `ts.run_worker()` processes.
  - `p2p-grpc` and `p2p-tcp`: `examples/train/p2p/worker/main.py`, with a driver and 2
    followers.
- **Size:** 3 partitions, 64 samples, batch 16, 2 epochs. This is deliberately tiny. It
  checks equivalence, not performance.
- **Runs:**
  - CPU with SGD (lr 0.05, momentum 0.9).
  - CPU with AdamW (lr 1e-3, wd 0.01).
  - A100 MIG 3g.40gb with AdamW.
- **Determinism:** `torch.use_deterministic_algorithms(True, warn_only=True)` in every
  process, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- **Pass threshold:** max |Δ| ≤ 1e-5 per epoch-mean loss, or ≤ 1e-4 for P2P because the
  P2P driver logs losses with 4 decimals.
- **Software:** torch 2.11.0+cu128, torchvision 0.26.0, transformers 5.19.0, peft 0.21.0,
  Python 3.13.

## Results (round 1, after the fixes below)

Raw data: `benchmarks/equivalence/results/results_{cpu_SGD,cpu_AdamW,cuda_AdamW}.json` and
console logs `out_*.txt`.

| Run | Cells passing | Worst max\|Δ\| local / gpipe | central (gRPC, TCP) | P2P (gRPC, TCP) |
|---|---|---|---|---|
| CPU, SGD | 53/54 | 4.5e-7 / 6.1e-7 | 0 (bit-exact) | 4.5e-5 (4-decimal logging) |
| CPU, AdamW | 53/54 ¹ | ~5e-7 | 0 on non-HF models; ≤2.4e-7 on HF | ≤5e-5 |
| A100, AdamW | 53/54 | ≤5.1e-7 | ≤2.4e-7 | ≤5e-5 |

The failing cell in each run is ResNet-18 + GPipe (see the next section).

¹ The gpt2-lora central-tcp cell failed on its first attempt because of issue 3 below. Rerun
alone after the fix, it matched the reference exactly. Both outcomes are recorded in the
JSON, under `rerun`.

The local modes differ from the reference by about 1e-7 because `LocalExecutor` reports
the loss in float32. The centralized modes report float64 and match exactly on CPU.

### Known, by-design difference: GPipe + BatchNorm

ResNet-18 with GPipe differs from the reference: max|Δ| is 7.7e-2 with SGD, 1.9e-2 with
AdamW on CPU, and 2.0e-2 with AdamW on the A100. GPipe splits each batch of 16 into 2
micro-batches of 8, so BatchNorm normalizes over 8 samples, and updates its running
statistics twice per batch. This is inherent to micro-batch pipelining; it is not a TorchSlicer bug. Models
without BatchNorm match under GPipe. **Paper/docs:** state it, and recommend GroupNorm or
LayerNorm, or a larger micro-batch, when exact equivalence matters.

## Issues found and fixed

Each issue has a regression test. Fix commit: `ac9f198`, except issue 4 (`c67f6f3`).

### 1. Tied weights trained as a different model (correctness, silent)

- **Symptom:** GPT-2 differed from the reference in all four distributed cells (central and
  P2P, gRPC and TCP), while untied models (Llama in this config, CNNs) matched. The exact
  pre-fix magnitudes were not kept: the logs were overwritten by later runs.
- **Root cause:** HF LLMs often tie the token embedding to the LM head, so a single
  `nn.Parameter` is used by both. With `pack`, the embedding sits in the first slice and
  the head in the last.
  - *Distributed/P2P:* each slice is serialized with `torch.save` separately, so each
    worker received its own copy, and the two copies then trained independently. The
    result is silently an untied model with twice the embedding parameters. This also
    affected the repo's own `examples/train/hf` GPT-2 example.
  - *Local:* the shared tensor was registered in two `SplitLayer` optimizers, so it was
    stepped twice per batch. That is harmless only for plain SGD without momentum, and
    wrong for momentum, Adam and weight decay.
  - Per-slice serialization also broke sharing *inside* a slice: two layers of the same
    slice that share a parameter ended up with separate copies.
- **Prevalence:** checked in the published `config.json` files.
  - Tied: GPT-2, BERT, Llama-3.2-1B, Qwen2.5-0.5B, SmolLM2-135M and Gemma-2.
  - Untied: Llama-3.1-8B, Mistral-7B, Qwen2.5-7B and Phi-3-mini.
  - In short, ties are common precisely in the small models that are realistic for split
    learning on edge clients.
- **Fix:**
  - **Ownership:** `core/tied.py` finds tied groups across partitions and makes the lowest
    partition index that uses a parameter its owner. Only the owner's optimizer contains it.
  - **Local:** all uses share one tensor, so its `.grad` already accumulates every use, and
    the owner steps it once.
  - **Distributed/P2P:** the coordinator (or P2P driver) sends each worker a list of tied
    groups (`SliceConfig.tied`) and an `init_token`. Workers re-tie the parameters inside
    their slice. In `_optimize`, each replica sends its gradient to the owner over the new
    `tied_grad` RPC and clears it. The owner waits for every replica gradient, adds them to
    its own, and steps. It then pushes the new value with `tied_update` before the batch
    completes, so the next batch sees identical weights everywhere. The `init_token`
    discards messages left over from before a recovery re-init.
- **Cost:** one parameter-sized message in each direction per batch, per tied group. Because
  the tied tensor is the embedding matrix, this message is large:
  - GPT-2-small: 50257 × 768 ≈ 38.6M params ≈ 154 MB in fp32.
  - Llama-3.2-1B: 128256 × 2048 ≈ 263M params ≈ 1.05 GB in fp32.
  - These numbers are computed from the configs, not measured. This cost belongs in the
    communication-cost section of the paper. Possible reductions: co-locate both ends on
    one worker (U-shaped split, M4), sync every k steps, send sparse gradients (only
    embedding rows that were used), or untie.
- **Verified by:** `tests/test_tied_weights.py` (in-process gRPC workers with Adam,
  compared with unsplit training), and the matrix (GPT-2 and gpt2-lora in all modes).

### 2. Workers trained with a float32-rounded learning rate (correctness, silent)

- **Symptom:** ResNet-18 with AdamW differed by about 3.3e-4 (central) and 3.9e-4 (P2P),
  even though every op was deterministic. With SGD the difference was invisible.
- **Root cause:** `OptimizerConfig.lr` in `worker_service.proto` is a `float` (32-bit). The
  coordinator wrote 0.001 into it and workers read back 0.0010000000474974513. We
  reproduced the drift exactly by training the reference with that rounded lr.
- **Fix:** `_build_optimizer_config` also puts the exact `lr` into `extra_params`
  (`torch.save`d), and workers use `{"lr": request.optimizer.lr, **extra}`, so the exact
  value wins. The proto field is kept for compatibility. The same fix went into the
  example workers `examples/train/p2p/worker`, `examples/train/lm/worker` and
  `examples/train/lm/lora_worker`.
- **Verified by:** `tests/test_runtime_control.py::test_optimizer_config_keeps_full_precision_lr`,
  and the matrix: central modes are bit-exact with AdamW on CPU.
- **Paper note:** this is a general pitfall for any SL framework that ships hyperparameters
  over a schema-typed RPC.

### 3. Worker init timed out for HF/LoRA slices (robustness)

- **Symptom:** `central-*` cells for HF models failed intermittently with "no result". The
  failure was more frequent under load and on GPU.
- **Root cause:** the coordinator's `init(SliceConfig)` RPC had a 5 s deadline. Deserializing
  a slice imports transformers/peft and materializes the weights, which takes longer than
  5 s on a busy host (and would take far longer for a real multi-GB slice).
- **Fix:** `StartupConfig.worker_init_rpc_timeout_s` default raised from 5 s to 300 s. A
  worker that isn't up yet still fails fast (UNAVAILABLE) and is retried. Only the wait for
  a live worker grew.
- **Verified by:** the gpt2-lora central-tcp rerun (footnote ¹) and the full A100 pass.

### 4. LocalExecutor dropped callback metrics (API)

- `train_epoch` returned a fresh `{"loss": avg}` instead of the callback-augmented
  `epoch_metrics`, so keys such as `val_acc` never reached `SlicedModel.train()` history.
  Fixed in `c67f6f3`, with the test
  `tests/test_local_executor.py::test_train_epoch_returns_callback_metrics`.

### Smaller fixes found along the way (`ac9f198`)

- A slice with nothing to optimize (only ReLUs, or only a tied parameter owned elsewhere)
  crashed; it now gets no optimizer.
- P2P could not run HF models; there is now a `get_pack()` hook in
  `examples/train/p2p/worker/main.py`.

## Harness pitfalls (not library bugs; for reproducibility)

- **Port collisions:** test ports in Linux's ephemeral range (32768–60999) were
  occasionally taken by outgoing connections. The harness now uses ports from 21000
  (`MATRIX_PORT_BASE`).
- **GPU determinism:** strict `use_deterministic_algorithms(True)` raises on some CUDA ops,
  so the harness uses `warn_only=True` in every process, workers included.

## Open (carried to M2, see `roadmap.md`)

- If a P2P follower crashes, the driver hangs, because P2P has no failure detection.
- `examples/train/lm/{worker,lora_worker}` duplicate the P2P driver and lack tied-weight
  support.
- The matrix should become a repo integration test, with a small subset in CI.
- Not yet covered: multi-host runs over a real network, bf16 autocast, gradient
  accumulation, checkpoint/recovery mid-run, and losses compared beyond epoch means (per
  batch, or final weights).

## Round 2 (M1 "split anything", 2026-10-15, branch `feat/m1-split-anything`)

M1 reworked how a model becomes a graph (`core/model_graph.py`) and how partitions are
validated (`strategies/base.py`), and added named cuts and cost strategies. Auditing the M0
tracer turned up three silent correctness bugs (5, 6, 8 below). Pre-fix numbers were
measured on `feat/m0-foundations` before any change, with the unsplit forward (eval mode)
compared to the chained slices, using `ts.slice(model, n=1)`.

| Model (M0) | M0 result |
|---|---|
| `F.relu` in the top-level forward | **silently wrong**: max\|Δ\| = 0.31 (the relu was dropped) |
| data-dependent `if` in forward (untraceable) | **silently wrong**: max\|Δ\| = 1.23 (fell back to a chain of children) |
| `x.view(x.size(0), -1)` | crash: shape `[-1]` flattened the batch dimension |
| `x + self.a(x)` at top level | crash: `_AddWrapper` got one argument |
| torchvision MobileNetV2, DenseNet-121 | crash: functional `adaptive_avg_pool2d` / `F.relu` dropped |
| torchvision ViT-B/16 | crash: dynamic reshape, `cat` of class token, `permute`, indexing dropped |
| HF GPT-2 / Llama without `pack=` | crash late: fallback chain `[transformer, lm_head]` fed a ModelOutput to the head |
| ResNet-18, `uniform`, n=6 | crash: autograd error, a slice started with `ReLU(inplace=True)` (issue 7) |

After M1, all of the traceable models above match the unsplit forward **exactly** (max\|Δ\| = 0) at
every valid cut (tests: `tests/test_split_anything.py`), and the untraceable models raise `ts.TraceError`.

### 5. Unrecognised functional ops were silently dropped from the graph (correctness, silent)

- **Root cause:** `_node_to_module` mapped only `torch.flatten`, `operator.add` and
  `view/reshape` to modules. Any other `call_function` or `call_method` node (`F.relu`,
  `torch.cat`, `x.permute`, `x.size`, `F.adaptive_avg_pool2d`, ...) returned `None` and was
  skipped. Its consumers were then wired to whatever came before. `view`/`reshape` also
  dropped non-constant shape arguments, so `view(x.size(0), -1)` became `reshape(-1)`.
- **Fix:** every live fx node becomes a graph node. Unknown ops become a generic
  `_FunctionWrapper`, which stores the target plus its args, with graph inputs replaced by
  `_Slot(i)`. Parameters read in forward become `_AttrWrapper`, which keeps the same
  `nn.Parameter`, so the tied-weight logic sees any sharing. A submodule called with extra
  constant arguments becomes `_CallWrapper`. The special wrappers (`_FlattenWrapper`,
  `_AddWrapper`, `_ReshapeWrapper`) are used only when they are exact, so graphs that were
  correct in M0 are unchanged: same nodes and edges for DNN, CNN, ResNet-18/50, TinyGPT
  (± LoRA), VGG-11, EfficientNet-B0, AlexNet and SqueezeNet (checked node by node). Nodes
  the output doesn't depend on are pruned. An unused in-place op raises `TraceError`.
- **Model input fan-out:** if the input feeds several nodes (`x + f(x)`), an explicit
  `input` node (`nn.Identity`) is inserted. Nodes reading the input are flagged
  `takes_input`, and validation keeps them in partition 0.
- **Verified by:** `test_every_valid_cut_reproduces_the_unsplit_forward`,
  `test_split_training_matches_unsplit`, and `test_distributed_traced_graph_matches_unsplit`.
  The last one ships the wrappers through `torch.save` to in-process gRPC workers.

### 6. Tracing failures silently fell back to a chain of children (correctness, silent)

- **Root cause:** `from_module` caught every exception and returned `from_sequential()`, a
  chain of `children()` in registration order. That is a different model whenever forward
  is anything but that chain.
- **Fix:** it raises `ts.TraceError`, which names the fx error and points to `pack=`
  (docs/tutorials/pack.md) and `expand=`. Several inputs, several outputs, and unused in-place
  ops also raise `TraceError`. `from_sequential` is still available, but is never used
  implicitly.
- **Behaviour change:** models that "worked" by accident under the fallback now need
  `pack=`. The fallback was only correct when forward was exactly the child chain, and fx
  traces those models anyway.

### 7. A slice starting with an in-place op crashed (robustness)

- **Root cause:** `SplitLayer` makes its input a leaf that requires grad
  (`detach().requires_grad_()`), and `ReLU(inplace=True)` on such a leaf is an autograd error.
  ResNet-18 hits this with plain `uniform` and n=6, and any fine-grained (`expand=`) split
  of a torchvision CNN hits it too.
- **Fix:** if a layer that reads the cut-point input is in-place (an `inplace=True`
  attribute, or a wrapped `*_` op), `SplitLayer` passes it `self.x.clone()`. Gradients
  still reach `self.x`. There is no cost for other slices.
- **Verified by:** `test_inplace_first_layer_of_a_slice_trains` and the fully expanded
  ResNet runs in the tutorials.

### 8. Validation accepted partitions that the executors run wrongly (correctness, silent)

- **Root cause:** `validate()` checked only multi-input nodes. A *single*-input node
  reading a node that was not the previous partition's last node passed validation. At
  runtime it silently received the previous partition's output instead. Partitions that
  were not contiguous were also accepted.
- **Fix:** partitions must be non-empty, contiguous and in order. Every edge that crosses
  a boundary must come from the last node of the previous partition, and only partition
  0 may read the model input. Errors list the valid cuts. `ModelGraph.valid_cuts()` gives
  the same rule to strategies.
- **Extension:** a multi-input node may now read the cut-point input. `Partition.predecessors`
  uses `-1` for it, and `SplitLayer` feeds it `x_in`. Without this, cutting right after a
  residual block would be impossible, because the next block's `add` reads that block's
  output. No proto change was needed (`int32`). Workers older than M1 would fail loudly
  on `-1`.
- **Verified by:** `test_validate_rejects_*` and
  `test_residual_add_right_after_a_cut_gets_the_external_input`, local and distributed.

### New features and how they were checked

- **Named cuts** (`cuts=[...]`, `NamedCutSplitter`). **`expand=`** traces into submodules,
  for finer cuts. **Cost strategies** `params`, `flops` and `profiled` choose the best
  min-max cut among the valid cuts. `balanced_cuts` matches a brute-force search on 200
  random cases. The DP breaks ties by the most even split.
- `flops` and `profiled` measure on a deep copy, under `torch.random.fork_rng`. Tests check
  that the model's state_dict, its grads and the global RNG are unchanged, so choosing a
  split never changes the training trajectory. Every strategy trains to the unsplit losses
  (≤1e-5) in tests.
- The run manifest records the node names and `expand`. `restore_model_from_run` reads
  `expand` back, so checkpoints of expanded graphs restore.

### Round 2 matrix (no-regression check)

Full CPU/SGD matrix rerun on the M1 code (`results/results_cpu_SGD.json` was overwritten; the M0 copy
was kept aside and compared): **all 54 cells, and both references per model, are bit-identical to M0**.
53/54 pass; the failing cell is still ResNet-18 + GPipe (BatchNorm, by design). All matrix models used `uniform`
over a graph that M1 leaves unchanged, so identical numbers are the expected outcome. The new
paths (generic wrappers, `-1` inputs, cost strategies, named cuts) are covered by the unit tests, local
and over in-process gRPC workers. Not rerun: CPU AdamW and A100 AdamW. The code paths are the same as with SGD.

### Open (carried)

- Recovery re-slices with its own divmod-uniform split, not the run's strategy or named
  cuts. With a fine-grained DAG graph it can land on an invalid cut and raise.
- `profiled` times each node in isolation, so cross-node fusion and transfers are not
  measured. Heterogeneous workers need per-worker scaling.
- `uniform` still ignores `valid_cuts()` (kept to leave existing partitions unchanged), so it
  can fail validation on fine-grained graphs where a cost strategy would succeed.

## Round 3 (M2 "P2P as a reference topology", 2026-10-15, branch `feat/m2-p2p`)

M2 moved P2P from the 694-line example script into the library (`ts.P2PExecutor`, `ts.run_p2p`).
The driver is a `DistributedExecutor` subclass whose worker 0 is an in-process `WorkerServicer`,
called directly (no loopback RPC, no serialization of the batch), with an in-process coordinator.
Everything else (servicer, transports, tied weights, heartbeats, recovery, run logging) is shared
with the centralized topology. The examples `examples/train/p2p/worker`, `examples/train/lm/worker`
and `examples/train/lm/lora_worker` now hold only a model, its data and a `ts.run_p2p(...)` call
(694 / 660 / 729 → 75 / 190 / 263 lines), so the LM examples gained tied-weight support and every
splitting option (`strategy`, `cuts`, `expand`).

**P2P equivalence is now checked bit-exactly.** The old harness parsed the driver's log, which
printed the epoch loss with 4 decimals, so P2P cells were compared at 1e-4. `run_p2p.py` now prints
the full-precision history like `run_central.py`, and every mode uses the same 1e-5 tolerance. On
the M2 code, DNN, TinyGPT and GPT-2 (tied weights) give max|Δ| = 0 against the unsplit reference
in `p2p-grpc` and `p2p-tcp` (CPU/SGD, logs `benchmarks/equivalence/logs`). The full matrix is under "Round 3 matrix" below.

### 9. Heartbeats wiped the workers' profiling stats (correctness of metrics, silent)

- **Symptom:** with `fault_tolerance.enabled`, `worker_epoch` rows (`profile.verbosity ≥ 1`)
  covered only the batches since the last heartbeat, not the epoch.
- **Root cause:** the heartbeat and the failure-report check call `get_stats(epoch=-1)` as a
  liveness probe, and `get_stats` always ends with `profiler.reset_epoch()`.
- **Fix:** `get_stats` with `epoch < 0` is a pure liveness probe: it returns `fatal_error` and
  touches nothing. Heartbeats now run in every distributed run (issue 10), so without this
  fix every run's profiling would have been truncated.
- **Verified by:** `tests/test_p2p.py::test_liveness_probe_does_not_reset_profiler_stats`
  (fails on the M1 code: `forward.count` = 0 after one probe, 1 after the fix).

### 10. A dead worker hung the driver/coordinator (robustness; carried from M0)

- **Symptom:** if a worker that is not the last one dies, the driver waits for `batch_done`
  forever. Reproduced on the M2 code with detection off (equivalent to M1): a 3-node P2P run
  where follower 1 stops after epoch 1 was still waiting when killed after 30 s. The same holds
  for the centralized topology unless `fault_tolerance.enabled`. If the *last* worker dies,
  sending it the labels fails, and the run ended with a raw `grpc.RpcError` after 5 s.
- **Root cause:** heartbeats, the only way to notice a crashed process, ran only with
  `fault_tolerance.enabled`, and the P2P example had none at all.
- **Fix:** new `fault_tolerance.detect_failures` (default true, env `DETECT_FAILURES`):
  heartbeats always run, and a dead worker raises `WorkerFailureError` (recovery still needs
  `enabled`). P2P inherits this, and `recover()` re-plans over the survivors. The driver's own
  slice is never moved to a follower, since that would send raw data off the driver, so a
  failure on the driver's slice ends the run with a clear error.
- **Cost:** one tiny `get_stats` RPC per worker every `heartbeat_interval_s` (5 s).
- **Verified by:** `test_p2p_dead_follower_fails_the_run_instead_of_hanging[0|1]` (≈1 s),
  `test_centralized_dead_worker_fails_the_run_instead_of_hanging`,
  `test_p2p_recovers_from_a_dead_follower`, `test_p2p_driver_slice_failure_stops_the_run`.

### Behaviour changes

- Raw inputs stay on the P2P driver: `test_p2p_driver_keeps_raw_inputs` checks that follower 1
  only ever receives 16-dim activations, never the 8-dim inputs.
- `P2PExecutor` accepts its peer list with or without its own address, and drops it, because the
  driver is always worker 0.
- The run manifest records `executor: p2p` for P2P runs (was `distributed` in the library, `p2p` in
  the old script).

### 11. `uniform` silently made fewer partitions than asked (correctness, silent)

- **Symptom:** `ts.slice(model, n=3)` on a 4-node graph returned 2 partitions (`[[0, 1], [2, 3]]`),
  so a 3-worker run quietly trained on 2 workers. The same happened for every (L, n) where
  `ceil(L / n)` sizes run out of nodes early: 1193 of the 1770 pairs with L < 60.
- **Root cause:** fixed slices of `ceil(L / n)` nodes, stopping when the nodes run out.
- **Fix:** only in those short cases, spread the remainder instead (`[[0, 1], [2], [3]]`), and
  raise if `n > L`. All pairs that already gave `n` partitions are unchanged (checked for
  every L < 60), so no existing split moves.
- **Verified by:** `tests/test_split_anything.py::test_uniform_gives_the_requested_number_of_partitions`.

### Capability-aware planning (feature, M2 block 2)

- **What:** `NodeInfo` now carries measured and declared capabilities: `compute_gflops`
  (a constant-matrix matmul benchmark that leaves the RNG untouched, or the declared
  `network.worker_compute_gflops`), `memory_mb` / `max_memory_mb`, `first_slice`, plus
  `rtt_ms` / `bandwidth_mbps` from the executor and `link_*` from each worker to the next one
  (`planning.measure_links`). Centralized workers send them when they register; any worker
  reports them through the new `describe` RPC, which is how a P2P driver learns them. `echo`
  and `probe_link` measure the network.
- **Deferred planning:** `ts.slice` still computes the partitions right away (local runs,
  `summary()`), but keeps the strategy. A distributed executor calls `SlicedModel.plan(nodes)`
  after discovery, and again in `recover()` for the survivors. Strategies whose `split` takes
  `nodes=` use the capabilities. The others are unchanged: the same count gives the same
  partitions.
- **Heterogeneous DP:** `balanced_cuts(..., speeds, mem, caps, cut_cost, link_cost)` minimises the
  largest `cost / speed + cut transfer time`, subject to each slice's estimated memory
  fitting its worker's cap. With every option off it does exactly the arithmetic it did
  before. Speeds within 25% count as identical, so benchmark noise on identical workers
  can't move the cuts.
- **Verified by:** `tests/test_planning.py`. The DP matches brute force on 300 random cases with
  speeds, caps and link costs. On 3 simulated P2P nodes (10/10/80 GFLOP/s) the fast node gets
  most of the parameters and training matches unsplit (≤1e-5). A fast node declaring a 5 MB
  cap gets one 4 MB layer instead of two, and still matches unsplit. Recovery re-plans with
  `params` for the survivors' speeds (`[[0], [1..6]]`, where the old uniform recovery gave
  `[[0..3], [4..6]]`). `first_slice` puts a centralized worker first. Links are measured
  between neighbours. Registration carries the new fields.
- **Behaviour change (fixes the M1 carried issue):** recovery now uses the run's strategy,
  given the survivors' capabilities. It falls back to the old uniform split only when the
  strategy can't produce that many partitions (for example, named cuts fixed for more workers).
- **Cost:** one `describe` per worker (the benchmark runs once per worker process: ~384³
  matmuls on CPU, 2048³ on GPU) and, with `planning.measure_network` (default on), 3 + 3 echoes
  of up to 1 MiB per worker at setup. Set `planning.heterogeneous: false` to plan exactly as before.
- **Limits:** the memory estimate counts parameters, gradients and two optimizer states
  (`memory_factor` 4), not activations. The speed is a single matmul number, which ignores
  memory bandwidth, kernels other than matmul, and the GPU's current load.

### Round 3 matrix (no-regression check)

Full CPU/SGD matrix rerun on the final M2 code. The M1 `results_cpu_SGD.json` was kept aside and compared.
**53/54 cells pass.** The failing cell is still ResNet-18 + GPipe (BatchNorm, by design, max|Δ| = 7.7e-2 as before).
- Every reference and all 36 local/GPipe/centralized cells are **bit-identical to M1**. Deferred planning, the probes,
  always-on heartbeats and the `uniform` fix change no loss (all matrix models use `uniform` with n=3 over graphs where
  it already gave 3 partitions).
- All 18 P2P cells (9 models × gRPC/TCP, now through `ts.P2PExecutor`) match the unsplit reference **exactly**
  (max|Δ| = 0, compared at full precision). They also agree with M1's 4-decimal P2P numbers.
Not rerun: CPU AdamW and A100 AdamW (same code paths as SGD).

## Round 4 (M3 "run anywhere", 2026-10-15, branch `feat/m3-run-anywhere`)

M3 makes one script + one config run on every target. `ts.run(model_fn, data_fn, **slice_kwargs)`
is the node entry point: `deploy.topology` (local / centralized / p2p) and the node's role
(`deploy.role`, env `ROLE`: all / coordinator / worker / driver / follower) pick what it does.
`ROLE=all` (the default) runs every node in one process, with workers on loopback through the
real gRPC/TCP stack. `python -m torchslicer launch SCRIPT -c CONFIG --target T` builds one node list
from the config (`deploy.plan_nodes`: names, roles, addresses, env) and either runs it
(`local`, `processes`: one OS process per node) or renders it (`compose`, `k8s`: Indexed Job +
headless Service, `baremetal`: one shell script per host). Guides: `docs/deploy/`.

**How it was checked.** `examples/run_anywhere/train.py` (SmallCNN, synthetic CIFAR-shaped data,
128 samples, batch 32, 3 slices, SGD momentum, 2 epochs, CPU) against the same model trained
unsplit in plain PyTorch: reference `[2.3475069403648376, 2.1740013360977173]`. After the fixes
below, every one of these gives exactly that (max|Δ| = 0):
local (`LocalExecutor`, history rounded to 6 decimals), `ROLE=all` centralized and P2P,
`--target processes` centralized (registration; static + TCP) and P2P (gRPC and TCP), and the
`--target baremetal` scripts run as written with every host = 127.0.0.1, both topologies, also with
the coordinator/driver started 8 s before its workers. `tests/test_deploy.py` (23 tests) keeps this.
Compose and k8s are **rendered, not run**: this container has neither Docker nor kubectl. The tests
check their structure (selectors match pod labels, subdomain = headless Service, peer list =
pod DNS names in index order, TCP tensor ports exposed, ConfigMap = the config file verbatim), and
their node env is the same `plan_nodes` output that the `processes` target runs.

### 12. `start_worker(run_config=)` ignored `network.device` (correctness, silent; M2 API)

- **Symptom:** on the GPU host, `ts.start_worker(run_config=cfg)` with `network.device: cpu` put the
  slice on CUDA, so an in-process simulation of a CPU config did not train on CPU. On the example
  above it gave `[2.3474976420402527, 2.174286723136902]` vs the CPU reference
  `[2.3475069403648376, 2.1740013360977173]` (max|Δ| = 2.9e-4; 5.7e-5 with plain SGD). This is CPU
  vs CUDA arithmetic, not a split error, but the run silently used a device the config did not ask for.
- **Root cause:** `WorkerServicer.__init__` resolves its device from the `DEVICE` env var only.
  `run_worker` works around that by exporting `DEVICE` from the config first; `start_worker` (added in
  M2 for in-process followers) did not.
- **Fix:** `start_worker` sets the servicer's device from `network.device` when the config names
  one (not `auto`); otherwise env/auto as before.
- **Verified by:** `tests/test_deploy.py::test_start_worker_honours_config_device` (fails before the
  fix with `'cuda' == 'cpu'`), and the losses above.

### 13. The local topology ignored `network.device` (consistency)

- **Symptom:** with the same config, local training ran on the device the model was built on (CPU),
  while workers resolved `network.device: auto` to CUDA, so "the same config" trained on two
  devices depending on the target.
- **Fix:** `ts.run` moves the model to `resolve_device(network.device)` in the local topology, so
  `network.device` means the same on every target. `LocalExecutor` itself is unchanged: a direct
  `ts.slice(...).train()` still uses the model's device.
- **Verified by:** `tests/test_deploy.py::test_local_topology_honours_network_device`.

### 14. With static discovery, the plan depended on which nodes were up first (correctness, silent)

- **Symptom:** a P2P driver (or a static-discovery coordinator) started before its followers probed
  them anyway. It logged `describe failed address=127.0.0.1:50052 error=StatusCode.UNAVAILABLE;
  planning without its capabilities` and planned with `gflops=0.0 mem_cap_mb=0` for the late
  followers. The run still trained (slice init retries 20 × 1 s), but a capability-aware strategy
  would have split differently depending on start-up timing, and a follower slower than ~20 s to
  start failed in init with no clear message. This is the normal case on compose and k8s, where
  pods start in any order and may still be pulling their image.
- **Root cause:** `StaticDiscovery` returns addresses, not live workers, and `_probe_nodes` made one
  attempt per worker.
- **Fix:** before probing, the executor waits up to `discovery.timeout` for each remote worker's
  channel to become ready. If a worker never comes up, the run fails with
  `worker <addr> not reachable within discovery.timeout=<t>s`. Cost: none when the workers are already
  up (one channel-state check each).
- **Verified by:** `test_late_followers_are_waited_for_and_planned_with_their_capabilities` (followers
  start 1.5 s after the driver; every node has `compute_gflops > 0` and training matches unsplit;
  fails before the fix) and `test_a_follower_that_never_comes_up_fails_within_discovery_timeout`.

### 15. The documented config priority was the opposite of the code (docs)

- `config.py`'s docstring, `CLAUDE.md` and `.env.example` said YAML > env vars. `RunConfig.load()`
  applies env vars on top of the YAML, so env wins (`EPOCHS=7` beats `epochs: 3`). Deployments rely
  on that: a node's `ROLE`, `PORT` or `WORKER_PEERS` must override the shared file. The docs now say
  Python kwargs > env vars > YAML > defaults. No behaviour change.

### Behaviour changes

- `ts.run_p2p` also takes the role from `deploy.role` (env `ROLE`: driver/follower) before
  `IS_DRIVER`, so the launcher can deploy the M2 P2P/LM scripts too. It also accepts `callbacks=`.
- New config section `deploy:` (`topology`, `role`, `name`, `image`, `namespace`, `gpus`) and env
  vars `TOPOLOGY`, `ROLE`. Version 0.5.0; console script `torchslicer` (= `python -m torchslicer`).

### Round 4 matrix (no-regression check)

Full CPU/SGD matrix rerun on the final M3 code, compared with M2's `results_cpu_SGD.json` (kept aside).
**53/54 cells pass.** The failing cell is still ResNet-18 + GPipe (BatchNorm, by design, max|Δ| = 7.7e-2 as before).
All 63 loss lists (references and every mode) are **bit-identical to M2**: the readiness wait, the `start_worker`
device fix and `run_p2p`'s role lookup change no loss. Not rerun: CPU AdamW and A100 AdamW (same code paths).

## Round 5 (M4 "SL settings", 2026-10-15, branch `feat/m4-sl-settings`)

M4 adds split-learning settings on top of the slice chain (`torchslicer/sl/`, tutorial
`docs/tutorials/sl_settings.md`):

- **Roles (`sl.setting`).** The *data owner* is the process that holds the data: the local
  process, the centralized coordinator, or the P2P driver. The setting decides which slices it
  hosts **in-process**:
  - `pipeline`: none (the centralized default, unchanged);
  - `vanilla`: slice 0 (the P2P default);
  - `u_shaped`: slice 0 and the last slice.

  The in-process hosting that the P2P driver had is now in `DistributedExecutor`, so a
  centralized coordinator can host slices too. The last slice gets its own gRPC server on
  `sl.tail_port`. `P2PExecutor` is now only defaults.
- **Cut hooks.** Each hook has `encode`/`decode` for activations and `encode_grad`/`decode_grad`
  for gradients, one method per side of a cut. Built in: `Float16`, `BFloat16`, `Quantize` (1/2/4/8
  bits, bit-packed), `TopK`, `GaussianNoise`, `Recorder`, `Chain`. A hook's payload is packed into
  one uint8 tensor, so real bytes go on the wire. Workers get their hooks in `SliceConfig`.
- **Metrics.** Every history entry and `metrics.jsonl` epoch row now carries:
  - `bytes_fwd`/`bytes_bwd` per cut, plus `bytes_input`, `bytes_label`, `bytes_client`,
    `raw_bytes_total`;
  - `compute_s` per slice, and client/server sums;
  - `sim_comm_s`/`sim_client_comm_s` for a configured link.

  Workers count what they send and return it in `get_stats`.
- **Multi-client sessions.** `ts.ClientSession` runs `sequential` (with a handoff `copy`/`keep`/
  callable) or `parallel` (with `on_round_end`, where FedAvg can plug in). It swaps the client
  slices' states between batches. This needs the client slices to be in-process, and the core
  never averages.

**How it was checked.** `tests/test_sl.py` has 53 tests, about 50 s. The model is a 3-slice MLP:
batch 16, 4 batches, SGD with momentum, 2 epochs, CPU.

- **Settings.** Each one trains exactly like unsplit PyTorch (within 1e-6) in every topology. The
  combinations tested are centralized pipeline/vanilla/u_shaped, P2P vanilla/u_shaped, gRPC and
  TCP, and `--target processes` (real OS processes) for P2P u_shaped + TCP and centralized vanilla.
  For each run, the byte metrics measured by the workers are identical, key by key, to what
  `LocalExecutor` counts for the same setting.
- **Byte counts.** The bytes are the closed-form values. Per cut: 4 × 16 × 16 × 4 B. Labels:
  4 × 16 × 8 B, only when they leave. A quantized payload has its exact packed size.
- **Hooks.** Distributed runs give the same losses (within 1e-6) and the same bytes as local runs.
  This holds with GPipe too. `GaussianNoise` is seeded per (seed, cut, batch, direction), so it is
  the same in every topology and leaves the global RNG alone.
- **Sessions.**
  - Sequential with `copy` handoff is bit-identical to one client on the concatenated data.
  - Parallel with one client is bit-identical to plain training.
  - Parallel with FedAvg at round end leaves equal client states.
  - Distributed sessions (centralized vanilla, P2P u_shaped; sequential and parallel) match local
    within 1e-6.
- **Privacy claim.** With vanilla, no worker ever receives a raw-input tensor, only cut
  activations. With u_shaped, no worker receives a label. A spy servicer checks both.
- **Recovery.** When a follower dies in a 4-slice u_shaped P2P run, the survivors re-plan to 3
  slices, and the driver still hosts the first and the last.

`examples/sl/train.py` is the end-to-end example, with `experiments/sl_u_shaped_p2p.yaml` (u_shaped,
8-bit cut 0, link model) and `experiments/sl_splitfed_local.yaml` (4 clients, parallel, FedAvg).

### 16. uint8 inputs crashed centralized workers (correctness; latent since M0)

- **Symptom:** on M3, a centralized run with uint8 inputs (e.g. raw image bytes) failed in its first
  batch: `WorkerFailureError: Worker 0 ... forward: shape '[8, 8]' is invalid for input of size 16`.
- **Root cause:** the coordinator's dtype table (`distributed._init_dtype_maps`) and the proto
  `DType` enum had no uint8. `_serialize_tensor` therefore labelled the uint8 buffer FLOAT32, and the
  worker read 64 bytes as 16 floats. P2P never hit this, because its driver feeds slice 0 in-process.
- **Fix:** `DType` gained `UINT8`, `INT8` and `BOOL` (M4 needs UINT8 for packed hook payloads anyway).
  Both sides map them.
- **Verified by:** `test_coordinator_sends_uint8_inputs` (losses equal the local run; `bytes_input`
  = 1 byte per value).

### 17. The last worker kept every label that arrived after its activation (memory, silent)

- **Symptom:** in `_forward_last`, a label that found its activation already waiting was used for the
  loss *and* stored in `_labels`, and nothing ever removed it. To reproduce, the label send was
  delayed past the activation. On M3, 4 batches left `labels left on the last worker: [4]`; on M4
  it is `[0]`. The loss is identical (1.437963). The dict grows by one label per such batch, for
  the whole run.
- **Root cause:** the label branch stored the label before checking for a cached output.
- **Fix:** `WorkerServicer._take_label` stores the label only when the activation hasn't arrived.
  It is shared by the gRPC path and by the new in-process `submit_label` (u_shaped).
- **Verified by:** `test_late_labels_are_not_kept_on_the_last_worker`.

### Behaviour changes

- **Proto.** New `DType` values (UINT8, INT8, BOOL), `SliceConfig.cut_hooks`/`n_slices` (17, 18),
  and `WorkerStatsResponse` communication fields (16–22). Workers must run M4 code. The Docker image
  needs a rebuild (it already needed one since M2). The coordinator stubs are unchanged
  (regenerated byte-identical with grpcio-tools 1.64.0).
- **Stats call.** The coordinator now calls `get_stats` on every worker once per epoch, even at
  `profile.verbosity: 0`. That call returns and resets the communication counters and, as before,
  the profiling stats. Liveness probes (`epoch=-1`) still reset nothing.
- **Compute time.** It is measured on every slice, every batch (perf_counter). On CUDA there is a
  synchronize at the end of each forward/backward phase. The next send would synchronize anyway,
  except after the optimizer step, so the cost is about one optimizer step per batch that can no
  longer overlap with the gradient copy.
- **History entries.** Distributed history entries now carry the SL metrics. Before, they held only
  `loss`.
- **Data-owner failure message.** "P2P driver's own slice failed" is kept for P2P. Centralized with
  vanilla/u_shaped says "the data owner (coordinator)'s own slice failed".
- **Node counts.** `discovery.n_workers` is the number of slices in both topologies. The data owner
  hosts its share, so a centralized vanilla run on 3 slices has 2 workers, and P2P u_shaped on 3
  slices has 1 follower. The launcher, `ts.run` and `run_p2p` count the same way. With u_shaped,
  rendered k8s exposes the tail port(s), bare metal gives it a free slot, and `processes` gives it
  a free port.
- **Version.** 0.6.0; new config section `sl:` and env vars `SL_*`.

### Round 5 matrix (no-regression check)

The full CPU/SGD matrix was rerun on the final M4 code and compared with M3's `results_cpu_SGD.json`
(kept aside). **53/54 cells pass.** The failing cell is still ResNet-18 + GPipe (BatchNorm, by
design, max|Δ| = 7.7e-2 as before). All 63 loss lists (references and every mode) are
**bit-identical to M3**. Moving the data owner's hosting into `DistributedExecutor`, the always-on
byte counting and compute timing, and the per-epoch `get_stats` change no loss. Not rerun: CPU
AdamW and A100 AdamW (same code paths). Test suite: 196 passed.

## Round 6 (M5 "out-of-core training", 2026-10-15, branch `feat/m5-offload`)

M5 adds `torchslicer/offload/` (survey and decision: `notes/offload.md`; tutorial
`docs/tutorials/offload.md`). `ts.OffloadExecutor` keeps every slice in a slow tier (pinned host
RAM, or files) and pages it into ≥2 device slots that are allocated once. The schedule is
L2L/Harmony:
- forward without a graph, keeping each slice's input and RNG state;
- the last slice runs forward and backward in one go;
- backward in reverse, recomputing each slice, with the optimizer step right after and an eager
  write-back.

The optimizer step runs on the GPU (state paged with the slice) or on the CPU (`auto` measures).
The new `memory` strategy picks the fewest slices that fit a slot budget.
`CostSplitter.node_memory` now counts activations when the strategy has an example input.

**How it was checked.**
- `tests/test_offload.py` (29 tests). Offload is bit-identical to `LocalExecutor` (losses and
  weights) for cpu/disk tiers, with and without GPipe, with dropout, BatchNorm (running stats
  updated once), tied weights, buffers shared by slices, and the config/env switch. On CUDA it
  matches local within 1e-5 for the GPU step, the CPU step, disk, and disk + `O_DIRECT`.
- A subprocess test caps the GPU at 100 MB. The in-memory executor runs out of memory; offload
  trains to the uncapped reference loss within 1e-5, under the cap.
- The equivalence matrix gained modes `offload`, `offload-gpipe` and `offload-disk`, plus
  `offload-cpuopt` on CUDA:
  - CPU/SGD: every model OK. ResNet-18 + GPipe differs exactly like plain GPipe (BatchNorm with
    micro-batches, by design).
  - CUDA/AdamW: every model OK, except two cases. ResNet-18 offload-gpipe is the same BatchNorm
    case. GPT-2 offload-cpuopt is SKIP: refused by design, because a trainable tied embedding is
    shared by two slices.
  - `offload-cpuopt` is compared with a new `reference-cpuopt`: unsplit, with the optimizer
    stepping on CPU copies. Against the plain CUDA reference, ResNet-18 differs by 1.1e-3, which is
    CPU-vs-CUDA AdamW rounding amplified as the loss falls from 2.4 to 0.2. Against the same
    arithmetic it matches at 3.3e-7.

**Performance** (A100 limited to a 6 GB cap; 666M-param GPT-shaped stack, AdamW,
batch 8 × 512; `benchmarks/offload/configs/gpt660m_b8.yaml`; one run of 6 steps, median of the
last 4):

| mode | s/step |
|---|---|
| unsplit, no cap (15.5 GB) | 0.61 |
| unsplit, 6 GB cap | OOM |
| offload, GPU step (4.0 GB peak) | 0.84 |
| offload, CPU step | 2.40 |
| offload, disk (page cache) | 3.92 |
| FSDP2 CPU offload | OOM |
| FSDP2 CPU offload + activation checkpointing | 5.10 |

The GPU-step mode is compute-bound. Compute (including the recompute) is 0.81 s; the transfer
bound is 0.47 s (10 GB in + 8 GB out at the measured pinned-copy bandwidth); the copies are hidden.

### 18. Restoring BatchNorm stats before backward broke cuDNN's saved tensors (dev, crash)
- **Symptom:** backward of a recomputed slice with BatchNorm on CUDA raised "one of the variables
  needed for gradient computation has been modified by an inplace operation".
- **Root cause:** cuDNN batch norm saves `running_mean`/`running_var` for backward, and the
  restore after the recompute bumped their version.
- **Fix:** `_replay` returns a `restore()` that is called after backward.
- **Test:** `test_batchnorm_stats_update_once`, `test_cuda_matches_local`.

### 19. Tensors shared by several slices were paged per slice (dev, silent NaN)
- **Symptom:** Llama in disk mode gave NaN losses.
- **Root cause:** HF Llama's rotary `inv_freq` buffer is one tensor used by 4 stages. Each slice's
  page re-pointed it, so a slice computed with another slice's home or slot view. Tied embeddings
  have the same shape of problem.
- **Fix:** any tensor used by more than one slice stays resident on the device. With
  `optimizer="cpu"`, a trainable shared tensor is refused.
- **Test:** `test_tensor_shared_by_slices_stays_resident`, the tied-weight tests, the
  llama/gpt2 matrix rows.

### 20. The cpu tier could be OOM-killed with no message, and the first check was wrong (robustness)
- **Symptom:** a 1.3B model's pinned homes (params + AdamW state ≈ 15 GB) hit the container's
  16 GB cgroup limit (`free` shows the host's 1.5 TB). The process was first throttled, which looked
  like a hang, then killed.
- **Fix:** `host_memory_available()` is cgroup-aware, and setup raises `MemoryError` with the
  numbers and points to disk mode. The first version of the check counted reclaimable page cache as
  used, and counted the pinned home twice, so it refused a 666M run that fits. It now counts clean
  file cache as available and the optimizer state (+ gradients) as new.
- **Test:** `test_cpu_tier_checks_host_memory`.
- **Note:** a DeepSpeed ZeRO-3 baseline launched without such a check OOM-killed the whole
  container. Baselines need the same estimate before they run.

### 21. Smaller issues found while building M5
- **Bandwidth probe:** it allocated 256 MB on the device and failed under a 100 MB cap. It now uses
  32 MB buffers, best of 3, and also measures both directions at once (~30 GB/s total here, not
  2 × 26).
- **O_DIRECT alignment:** pinned buffers from the caching host allocator are often not
  page-aligned (offsets of 512 or 1536 B), which gave EINVAL. Fixed with `_host_aligned`.
- **Meta placeholders:** `Parameter.data = meta tensor` is refused, so evicted tensors on disk
  point at mmap views of their files.
- **GPU OOM on this machine:** with no NVML access, a capped allocation fails an NVML internal
  assert instead of raising `OutOfMemoryError`. The test and the benchmark treat that assert as
  OOM.

### Measurements that changed the design
- **Disk I/O path** (666M model, s/step):
  - `mmap` + memcpy: 11;
  - O_DIRECT `pread`/`pwrite` (16 MiB × 4 threads): 12–16;
  - buffered `pread`/`pwrite`: 3.9.

  Standalone, O_DIRECT reaches 4.5 GB/s for writes and 15 GB/s for reads (the host caches below
  O_DIRECT), but under training's mixed traffic only ~1.4 GB/s in total. So `direct_io` is an
  option, off by default. The 666M disk run fits in the page cache, so it measures RAM. A model
  larger than RAM (the 1.3B config) is not run yet.
- **CPU optimizer step:** the container has a 4-CPU quota. For 666M params, fused CPU AdamW
  takes ~2.3 s, against ~11 s for foreach. Both are slower than the transfers they save, so `auto`
  picks the GPU here.
- **Micro-batch order:** offload runs slice-major and GPipe micro-batch-major. With dropout in
  several slices and M > 1, the masks differ from GPipe's: same distribution, not bit-identical.
  Documented; with M = 1 everything is identical.

### Behaviour changes
- `CostSplitter.node_memory` adds the measured activation bytes when the strategy has an
  `example_input` (flops, profiled, memory). This affects distributed planning only with memory
  caps.
- `LocalExecutor` gains an `_epoch_extras()` hook for executor-specific history fields.
- Config: new `offload:` section (`mode`, `optimizer`, `dir`, `slots`, `device`, `direct_io`) and
  `OFFLOAD*` env vars. Version 0.7.0.

### Open (carried)
- DeepSpeed ZeRO-3/ZeRO-2 baselines (they don't fit in 16 GB of host RAM at 666M); the 1.3B
  disk run (larger than RAM); the batch-32 GPipe config; mean ± std over repeats.
- `tests/test_sl.py::test_distributed_hooks_match_local[hook0]` failed once in a full-suite run
  and passed in 5 reruns. It looks flaky under load (distributed, gRPC); not touched by M5.
- The full matrix was not rerun end to end on the final code. The offload modes and touched
  models were rerun after each fix. Test suite: 221 passed, plus the flaky test above.

## Round 7 (M6 "NAT-friendly transport", 2026-10-15, branch `feat/m6-nat`)

M6 lets any node work behind NAT, so it needs only outbound connections
(`torchslicer/transport/tunnel.py`; tutorial `docs/tutorials/nat.md`):

- **Tunnel.** A node behind NAT (`transport.nat`) opens no port. It dials each node that has to
  call it, opens one bidi gRPC stream there (`TunnelService.open`, added to
  `worker_service.proto`) and announces its addresses (HELLO). The far end then sends its
  unary calls to that node down the stream; the NAT'd node's own calls go out directly as
  before. A tunnel carries the serialized messages gRPC would send, so cut bytes, hooks and
  metrics are unchanged.
- **Router.** Each node has one. `router.channel(addr)` replaces `grpc.insecure_channel`, so
  every generated stub and call site is unchanged. Each call is routed when it is made:
  in-process service, inbound tunnel, relay, or a plain (now cached) channel. Raw TCP is used for
  a peer only while it is directly reachable.
- **Who dials whom.** The NAT'd data owner opens tunnels to every remote worker after
  discovery. A NAT'd worker opens one to its coordinator/driver at start, and to its neighbours
  and tied-weight peers at `init`.
- **Relay.** `python -m torchslicer relay` is a router that forwards by target address. Every
  node with `transport.relay` dials only the relay. That covers two NAT'd neighbours, and a NAT'd
  coordinator with `CoordinatorDiscovery`.
- **Launcher.** `transport.nat` in a shared config means the data owner (`plan_nodes` sets
  `TRANSPORT_NAT` per node). `relay: auto` adds a relay node (processes, compose, k8s Deployment,
  bare metal), and `ROLE=all` starts one on loopback.

**Deviation from the plan.** The roadmap said "the gradient comes back in the reply". A
reply-only design covers one of the reverse calls: the gradient to slice 0. A NAT'd client also
receives the U-shaped activation for its tail slice, `report_metrics`, `report_worker_error`,
tied-weight gradients and updates, heartbeats and its own `init`. So every call to a NAT'd node
travels back on the connection it opened: the gradient *is* in a reply on that connection, and
the rest come back the same way. Per-call routing kept the executors' and workers' send code as
it was.

**How it was checked.**
- `tests/test_nat.py` (24 tests, about 80 s); the 3-slice MLP from `test_sl.py`, 2 epochs, CPU.
  - Data owner behind NAT: P2P vanilla (gRPC, TCP), P2P u_shaped (TCP), centralized vanilla
    (gRPC) and u_shaped (TCP). Losses match unsplit PyTorch within 1e-6. They equal the direct
    run exactly, as do the byte metrics, key by key.
  - Through a relay: P2P u_shaped, centralized pipeline (TCP configured, so gRPC through the
    relay) and centralized vanilla; same checks.
  - The NAT'd driver has no gRPC server, no tail server and no TCP listener, so any call
    that dialed it would fail. The batches complete, its router counts at least 16 calls arriving
    by tunnel (8 backward + 8 `report_metrics`), and the follower next to it counts the calls it
    sent by tunnel.
  - NAT'd workers (centralized pipeline: workers 0 and 2 behind NAT, worker 1 reachable,
    CoordinatorDiscovery, gRPC and TCP) register, train and match the reference.
  - Two adjacent NAT'd workers without a relay fail at `init` with a message that names the
    relay. With a relay, a 4-slice U-shaped run where the driver and both (adjacent) followers
    are NAT'd matches the reference.
  - Tied weights across a NAT'd driver (embedding on slice 0, head on the last follower) match
    local.
  - A NAT'd follower that dies fails the run with `WorkerFailureError` within the timeout,
    instead of hanging. Recovery with a NAT'd driver re-plans over the survivor.
  - Tunnel errors keep their gRPC codes: UNKNOWN with the remote exception, DEADLINE_EXCEEDED,
    and UNAVAILABLE (fast, never dialed) once a NAT'd address's tunnel is gone.
  - `--target processes` (real OS processes): P2P u_shaped with the driver behind NAT (TCP),
    the same through a relay process, and centralized vanilla + static discovery through a relay.
    All match `--target local` within 1e-6. Plan, compose and k8s rendering of the relay are
    checked for structure.
- The tutorial's code blocks run in `tests/test_docs.py`.

### Round 7 matrix (no-regression check + 2 new modes)
`run_matrix.py` (CPU, SGD) gained `p2p-nat` (P2P driver behind NAT, TCP) and `p2p-relay` (driver
and followers behind NAT, a relay process, gRPC), as real OS processes. Full run on the final
code: **97/99 OK**. Both new modes are bit-exact (max|Δ| = 0) on all 9 models. The 2 DIFFs are
ResNet-18 + GPipe and ResNet-18 + offload-gpipe (7.7e-2), the known by-design BatchNorm
difference. `results/results_cpu_SGD.json` was rewritten by this run.

### Cost of the tunnel and the relay (`benchmarks/nat/bench_nat.py`)
U-shaped P2P, driver + 1 follower as OS processes on loopback, CPU, 1 thread per process. A
3-slice MLP whose cuts carry 1 MiB per batch and direction (batch 256 × 1024 fp32), so 4 MiB
cross per batch and the driver receives 2 MiB of them. 32 batches per epoch, 3 epochs; time per
batch = mean of epochs 2–3. Mean ± std over 3 runs; losses are identical in every mode
(`benchmarks/nat/results/bench_nat.json`).

| mode | ms/batch | vs direct |
|---|---|---|
| direct, gRPC | 53.0 ± 5.1 | — |
| direct, TCP | 38.3 ± 0.3 | — |
| driver behind NAT, gRPC | 51.8 ± 3.1 | same within noise |
| driver behind NAT, TCP | 48.7 ± 2.8 | +10.4 ms (+27%) vs direct TCP |
| everything through a relay | 58.7 ± 3.6 | +5.7 ms (+11%) vs direct gRPC |

- **The tunnel itself costs nothing measurable.** It replaces a gRPC call with a gRPC stream
  message.
- **With TCP, the reverse traffic is the cost.** The driver's 2 MiB per batch come back over the
  gRPC tunnel instead of raw TCP. A raw-TCP reverse path is possible (the peer would write frames
  back down the socket the client opened) but is not built; see Open.
- **The relay adds one hop per message.** On loopback that is a copy; over a WAN it is a second
  link.

### 22. A NAT'd worker could lose the Ack of its own shutdown (dev, spurious warning)
- **Symptom:** in the all-NAT relay test, the driver logged `shutdown failed worker=nat-0:1 ...
  UNAVAILABLE` for both NAT'd followers in 3 out of 3 runs. Training was unaffected.
- **Root cause:** `shutdown()` returns its Ack and stops the server from a thread. A gRPC server
  stops with `grace`, which lets the reply go out. The NAT'd worker's stand-in (`_NoServer`)
  closed its tunnels at once, sometimes before the Ack frame was sent.
- **Fix:** `_NoServer.stop(grace)` waits up to `grace` (at most 1 s) before closing, like
  `grpc.Server.stop`. After the fix: 0 warnings in the same test.
- **Cost:** shutdown of a NAT'd worker takes up to 1 s longer.
- **Test:** checked by hand on `test_every_node_behind_nat_through_a_relay` (log count). There is
  no assertion, because the symptom is only a log line.

### Behaviour changes
- Every stub is built on the node's `Router`. Channels are now cached per address per node; before,
  there was one channel per stub.
- Raw TCP vs gRPC is now decided per send (`router.is_direct(peer)`) instead of at `init`. Workers
  always build gRPC stubs for their neighbours, and also TCP clients when `transport.tensor: tcp`.
- The coordinator's gRPC pool is `max(8, 2n + 4)` (was `max(4, n)`): each NAT'd worker's tunnel
  holds one server thread.
- `run_worker` passes the advertised address to `start_worker`; `WorkerHandle.stop()` also closes
  the worker's tunnels.
- `plan_nodes` sets `COORDINATOR_ADDRESS` on every worker/follower (a NAT'd one tunnels to it).
- `announce_to_coordinator(channel=)`. Config: `transport.nat`, `transport.relay`,
  `transport.tunnel_keepalive_s` (env `TRANSPORT_NAT`, `TRANSPORT_RELAY`, `TUNNEL_KEEPALIVE`).
  New command `python -m torchslicer relay`. Version 0.8.0.

### Open (carried)
- **Raw-TCP reverse path.** With `transport.tensor: tcp`, traffic to a NAT'd node goes over the
  gRPC tunnel; that is the +10 ms/batch above.
- **No hole punching (STUN/ICE).** Two NAT'd peers always go through the relay.
- **No TLS or authentication.** That covers tunnels and the relay, as for every channel. Any node
  can announce any address to a relay, which on an untrusted network lets it intercept that
  address's calls. Needs auth before real deployments. This is milestone M7 (security).
- **No real NAT tested.** Everything ran on loopback, where "behind NAT" means "listens on
  nothing". A run across a real NAT (and a WAN relay) belongs with M13 (real environment
  and testbed; renumbered in the 2026-10-15 roadmap review).
- **The keepalive interval (20 s) is not measured** against real NAT timeouts.
- **Router stats are approximate.** `stats` counters are incremented without a lock; they are
  diagnostics only.

---

## M9: Fast Tensor Transport (2026-10-16, branch `feat/m9-fast-transport`)

### What was added

Three new tensor-transport modes on top of the existing `grpc` and `tcp` paths:

1. **`tcp` — pinned-memory async D2H** (`_TensorPeerClient`):
   Pre-allocate one pinned host buffer per connection (high-water-mark, 25% headroom
   on growth). `tensor.to(pinned, non_blocking=True)` + `torch.cuda.Event` — the D2H
   copy and frame-header construction overlap; `event.synchronize()` fires only when
   `sendall` needs the bytes. Non-CUDA tensors fall through to the original path.

2. **`ipc` — CUDA IPC zero-copy** (`_IpcPeerClient` / `_read_ipc_frame_rest`):
   Sender: `ForkingPickler` serialises the GPU tensor to an ~350-byte IPC handle;
   handle travels over a TCP socket (kind byte `b"f"`/`b"b"`); receiver reconstructs
   via `pickle.loads`, clones, sends 1-byte ACK. Same-machine check via `_is_same_machine`
   (localhost, 127.x, hostname resolution); cross-machine falls back to `_TensorPeerClient`.
   `_RecvBuf` (high-water-mark `bytearray`) avoids per-frame allocation on the receive side.

3. **`nccl` — NCCL P2P isend/irecv** (`_init_nccl_group` / NCCL sentinel path):
   Coordinator starts a `torch.distributed.TCPStore` master at `coordinator_port + 10`.
   Workers call `_init_nccl_group` at `init` time: `TCPStore(is_master=False)` → `init_process_group`.
   On forward/backward, sender calls `dist.isend` + sends gRPC metadata with empty `data`
   (NCCL sentinel); receiver detects empty `data` + NCCL mode → `dist.recv` from the
   appropriate rank. `dist.destroy_process_group` called before re-init for persistent workers.

### Bugs found and fixed

- **`_handle_tensor_conn` kind-byte dispatch**: original code assumed uppercase "F"/"B";
  IPC uses lowercase "f"/"b". Fix: read 1 kind byte, dispatch on both cases, share the
  `_RecvBuf` per connection for the TCP path.
- **`start_tensor_server` guard**: was `if self._tensor_transport != "tcp": return`. IPC
  also uses the TCP server (sends handles, not raw tensors). Fix: guard changed to
  `not in {"tcp", "ipc"}`.
- **NCCL sentinel detection**: the check `not request.input.data` is true for an empty
  bytes field, which is indistinguishable from a zero-byte tensor in other paths.
  Fix: three-way guard — `transport=="nccl" and dist_group is not None and not data`.
- **`init` peer-client creation**: the `ipc` branch also needs a `_TensorPeerClient`
  as fallback (for cross-machine and CPU tensors). Fix: always construct `_TensorPeerClient`
  first, then wrap in `_IpcPeerClient` for same-machine CUDA.

### Test coverage

- All 272 existing tests pass after M9 changes (no new failures).
- The pinned-memory path is covered by the existing TCP tests and matrix modes (the send/receive
  logic is unchanged; only the D2H timing differs).
- **Matrix modes `central-ipc` and `p2p-ipc` (2026-10-17).** Every node is a real OS process, so
  IPC crosses CUDA contexts for real. 9 models x 2 modes x 2 devices = 36/36 OK, SGD:
  - CPU: bit-exact (max|Δ| = 0). This is the TCP fallback, since CPU tensors never take IPC.
  - CUDA (A100 MIG): dnn, cnn, resnet18, tinygpt and tinygpt-lora are bit-exact. gpt2, llama and
    their LoRA variants are within 1.2e-7 to 4.8e-7. That is GPU nondeterminism: the recorded CUDA
    `local` mode differs by as much (2.4e-7 to 4.5e-7, `results_cuda_AdamW.json`).
  - Checked that IPC is really used, not a silent fallback: a probe that counts sends per client
    class saw only `_IpcPeerClient` sends on CUDA tensors (cnn: 8/16/8 per node, 0 TCP).
- `central-nccl`/`p2p-nccl` are in the matrix as opt-in (`MATRIX_NCCL=1`). They are
  bit-exact on CPU (gloo). On one GPU they now stop at init with the reason (issue 23). Real
  NCCL needs >= 2 GPUs: not run yet.

### Known limitations (carried)

- ~~IPC incompatible with cut_hooks~~ **wrong, corrected 2026-10-17**: it worked through
  the TCP fallback (payloads were packed on the CPU); now IPC carries them (issue 24).
- ~~NCCL incompatible with cut_hooks~~ it was broken (issue 25); fixed 2026-10-17.
- **NCCL requires all-CUDA workers or all-CPU ones** (CPU = gloo). Mixed devices are not
  supported.
- **NCCL not validated**: no equivalence or benchmark run on this hardware (single MIG GPU).
  IPC is validated (see Test coverage).
- **NCCL persistent-worker re-init**: `destroy_process_group` is called inside
  `_init_nccl_group`. If the coordinator re-uses the same port across runs the TCPStore
  may not be ready immediately; a 60-second timeout is set.

### Pinned-memory improvement: no measurable benefit on loopback (2026-10-16) — **conclusion wrong, see the 2026-10-17 corrections**

A direct comparison (TCP_NO_PINNED=1 to force the old sync path) found no statistically
significant difference between pinned-async and sync-cpu D2H across four model sizes on
loopback. Differences ranged –6% to +6%, all within run noise. The implementation is kept
(correct, low overhead, sound for future multi-node use) but the "1.5–2× speedup" claim
applies to TCP vs gRPC, NOT to pinned vs sync. The roadmap target of 1.5–2× for pinned
over bare TCP is not met in the loopback benchmark; it remains a hypothesis for
bandwidth-limited scenarios (slow PCIe, many concurrent slices).

### NCCL deadlock on sequential SliceConfig init (2026-10-16) — fixed

`_init_slices` sends SliceConfigs sequentially. Rank 0's `init()` called `dist.init_process_group`
synchronously, blocking until all world_size ranks joined. Rank 1 never received its SliceConfig
because the coordinator was stuck waiting for rank 0's `init()` to return. Classic deadlock.
Fix: `_init_nccl_group` now submits the rendezvous to a background thread and stores a Future;
`_wait_nccl()` is called lazily before the first actual NCCL transfer.

### NCCL hangs on MIG single-instance (2026-10-16) — known limitation

After fixing the sequential deadlock, NCCL still hangs during `init_process_group` when both
processes land on the same MIG GPU instance. NCCL's communicator setup probes NVLink/PCIe P2P
between what it identifies as the same device, which deadlocks in the MIG context. NCCL requires
genuinely separate GPU devices (multi-GPU node or multi-node). Not a bug in TorchSlicer;
documented in `docs/tutorials/transport.md` and experiments.md.
**Cause wrong, see issue 23**: NCCL fails cleanly ("Duplicate GPU detected"); the hang was ours.

### Corrections and fixes (2026-10-17)

Found by measuring each piece of the M9 benchmark separately (scratchpad micro-benchmarks,
then the rewritten `benchmarks/transport/bench_transport.py`). Pre-fix numbers recorded first.

**Benchmark errors (behind every M9 number logged on 2026-10-16):**
- *"Act. MB/cut" was per epoch, not per batch.* It came from `bytes_fwd` of a whole
  epoch (16 batches). Real cuts: small 0.07, medium 2.1, large 0.52, xlarge 3.15 MB per batch,
  not 1/33.6/8.4/50.3 MB.
- *ms/batch included setup.* It was the whole `train()` time (discovery, probing with its echo
  bandwidth test, init, CUDA warm-up) over all batches; the "exclude first epoch" comment was
  not implemented. IPC/medium: 22.3 ms reported vs 4.4 ms in steady state. This was the
  "~21 ms fixed cost".
- *Thread oversubscription.* torch sees the host's 40 cores (32 intra-op threads), but the
  container's quota is 4 CPUs. Every CPU tensor op on the gRPC/TCP receive path woke 32
  spinning OpenMP threads: about 15 CPU-seconds per process per run, and the cgroup was
  throttled in 74 periods vs 4 with IPC. That also inflated the slices' measured compute
  (580 ms vs 18 ms per epoch for the same layers). With `OMP_NUM_THREADS=4`, medium: gRPC
  110 -> 22.5 ms/batch, TCP 58 -> 13 ms, IPC ~5 ms. The benchmark now sets threads to the
  quota in both processes. The framework does not do this yet (open item).

**23. NCCL on one GPU hung instead of failing.** Symptom: `p2p-nccl` with 2 slices on one
GPU hung until the 240 s matrix timeout. Root cause: NCCL refuses two ranks on one GPU
(`ncclInvalidUsage: Duplicate GPU detected`, checked with a 2-process script, NCCL 2.28.9).
But PyTorch creates the communicator lazily, so the error came at the first send, inside a
worker thread. And `_wait_nccl` swallowed rendezvous errors and switched that worker alone to
gRPC, leaving its peer blocked in `dist.recv`. The 2026-10-16 explanation (NVLink probing on
MIG) was wrong. Fix:
- each rank publishes `hostname/GPU-UUID` in the store and refuses duplicates with a clear
  reason;
- eager communicator init (`device_id=`) and a per-init `PrefixStore`;
- each rank writes `ok`/`err:<reason>` under `ts/nccl/<init_token>/status/<rank>`. The
  coordinator polls these after `_init_slices` (`_wait_nccl_ready`) and raises on the first
  error;
- `_wait_nccl` raises instead of falling back; the store master failing to start raises.

Now stops in 5–8 s with "needs one GPU per slice ... use transport.tensor=ipc on one GPU".
Test: `tests/test_transport.py::test_nccl_on_one_gpu_fails_fast_with_the_reason`.

**24. Hooked payloads never used IPC.** `pack_payload` moved every tensor to the CPU, and
`_IpcPeerClient` sends CPU tensors over TCP. So IPC + hooks was correct (identical losses to
TCP + hooks), but every send went over TCP: a send probe counted `ipc:cpu` -> TCP fallback
on all sends. Fix: `pack_payload` keeps the payload's device. `unpack_payload` copies only a
4 KB header peek to the host and slices the buffers where they live (with a full-copy
fallback for longer headers). After: all sends `ipc:cuda:uint8`, same losses. Tests:
`test_pack_payload_stays_on_the_payload_device`, `test_unpack_payload_header_longer_than_the_peek`,
`test_ipc_matches_tcp_on_gpu[hooks1]`.

**25. NCCL receive paths skipped hook decoding.** The three `dist.recv` sites (forward,
last-slice forward, backward) passed the received tensor straight to the stage, skipping
`_decode_in`/`_decode_grad`. Pre-fix symptom (gloo, CPU, quantize-8): `RuntimeError: Input
type (unsigned char) and bias type (float) should be the same`. On CUDA the NCCL send of the
CPU-packed payload would have failed first. Fix: decode after each `dist.recv` (+ issue 24
for GPU payloads). After: matches TCP + hooks exactly. Test: `test_nccl_path_decodes_cut_hooks`.

**Pinned memory, re-measured.** The old path (`.cpu()`) allocates fresh pageable memory per
send. Above glibc's 32 MB mmap threshold every send page-faults: 20.3 ms vs 1.4 ms pinned at
33.6 MB, 0.77 vs 0.36 ms at 8.4 MB. In the benchmark (mean of 3): no difference up to 8 MB
cuts, **1.8x faster at 32 MB (92.5 -> 51.3 ms/batch) and 64 MB (168 -> 95 ms)**. The
2026-10-16 "no benefit" came from cuts of 0.07–3 MB.

**Coverage.** `tests/test_transport.py` (8 tests, real OS processes). The `nccl` transport is
tested on CPU via gloo: same TorchSlicer code path, and a probe confirmed every transfer used
`dist.isend`/`recv`. Matrix `central-nccl`/`p2p-nccl` (opt-in `MATRIX_NCCL=1`) are bit-exact
on CPU. Full CPU matrix after these fixes: 115/117, the 2 known GPipe + BatchNorm cells;
no existing cell's losses changed. **Still unverified: real NCCL on 2+ GPUs.**

### 26. gRPC collapse above 32 MB: page faults on fresh buffers (2026-10-17)

Symptom: gRPC took 61 ms/batch at 8.4 MB cuts but 724 ms at 33.6 MB (12× for 4× the
bytes). Cause, measured stage by stage (one transfer, A100 host, 4 threads): every fresh
buffer of 32 MB or more page-faults on first touch, at about 0.6 ms/MB. Each transfer makes
about 9 full-size copies (D2H, `tobytes`, protobuf serialize, gRPC core, parse, `msg.data`,
`bytearray`, `clone`, H2D). Each copy is ≥10 GB/s below the threshold and about 1.6 GB/s
above it: `deserialize_tensor` 6.8 ms at 16 MB, 74 ms at 32 MB. Two mechanisms are
involved:
- glibc serves allocations ≥32 MB with a fresh `mmap`;
- gRPC handler threads allocate from secondary malloc arenas. Their heaps are capped at
  64 MB and trimmed back to the OS, so `MALLOC_MMAP_MAX_=0` alone fixed 32 MB (740 -> 268
  ms) but not 64 MB (1423 -> 1393).

Fix (code): `deserialize_tensor` views the already-copied `msg.data` (upb copies on access)
and makes one `.to(device, copy=True)` straight to the slice's device, dropping the
`bytearray` and `clone` copies. Micro: 74 -> 33 ms at 32 MB on the GPU path. The send side
can't be fixed the same way: protobuf only accepts `bytes`, and every new `bytes` ≥32 MB
page-faults however it is built.

End to end (grpc, ms/batch, mean of 2):

| cut | before | deserialize fix | + `MALLOC_MMAP_MAX_=0 MALLOC_ARENA_MAX=1` |
|---|---|---|---|
| 33.6 MB | 740 | 654 | 175 |
| 67 MB | 1423 | 1276 | 348 |

The malloc settings restore linear scaling, and TCP without the pinned buffer matches pinned
(49 / 97 ms). They are process-wide: RSS keeps its peak, and one arena means
allocator lock contention. So they are documented, not set by the framework (open: an
opt-in knob). The rest of the gap to TCP is gRPC's own copies; removing it needs a
streaming/raw-bytes RPC (not done).

Verification: tests 278/280. The 2 failures are `test_offload.py::test_cuda_matches_local`
(NaN). It is flaky with and without this change (stash check: 2/4 and 3/4 failing runs),
while another tenant held the GPU at 99 %. Possibly a timing race in offload, open.
CPU matrix 115/117 (the 2 known GPipe + BatchNorm cells); every cell's losses
bit-identical to before the change.

## M11: Data and Scale Simulation (2026-10-17, branch `feat/m11-data-scale`)

### Bugs found and fixed

**27. After aggregation the last client of a round kept its old weights (M8, silent wrong
result).** Found while designing evaluation for sessions. The executors replaced
`session.states` with the `on_round_end`/`aggregate` output, but the live client slices still
held the active (last) client's pre-aggregation state. The next round's first client switch
saved that stale state over the aggregate. Probe: with SplitFed V2 and 3 parallel clients,
before round 2's first switch all three states equal the average. After it, client 2's
differs. So client 2 never received the average: one of N clients was left out of FedAvg
every round. It affected local and distributed sessions with `ts.SplitFedAlgorithm`,
`SplitFedV1Algorithm` or any `aggregate`/`on_round_end`, since M8 (0.8.0). Fix:
`ClientSession.replace_states(states, adapter)` also loads the active client's new state into
the slices; both executors use it.

**28. After `fedavg` every client inherited the previous client's optimizer state (M8,
silent wrong result with momentum/Adam).** `fedavg` drops optimizer state ("re-warms"), but
`SplitLayerClientState.set_state` skipped the optimizer when the state had none. That left the
live optimizer's buffers, i.e. the momentum of whichever client ran last (and V1's server
copies likewise). Fix: no optimizer state now means `optimizer.state.clear()`.
Test for 27 + 28: `test_algorithm.py::test_splitfed_matches_reference[V2, V1]`, which
compares against SplitFed written in plain PyTorch (3 clients, SGD momentum 0.9, 2 rounds) at
1e-6. Both fail without the fixes (round 2) and pass with them.

**29. The distributed executor silently ignored most `ts.Algorithm` hooks (M8, silent wrong
result).** It ran only `configure`, `on_round_end` and `aggregate`. A custom `loss`, local-loss
training (`backward_policy`, `client_loss`, `step_aux`) and SplitFed V1's per-client server
copies were dropped, and the run trained plain SplitNN/SplitFed V2 instead. Also,
`select_clients` was called by no executor at all. M8's "one object, all executors" didn't
hold. Fix for now: the executors refuse what they don't run, with a clear `ValueError`
naming the hooks (`algorithm.base.check_supported`; a non-FULL backward policy is checked per
round). Implementing them in the distributed executor is open (scheduled with M10).
`select_clients` (client sampling) belongs in M11's many-client simulation. Tests:
`test_sl.py::test_distributed_refuses_algorithm_hooks_it_does_not_run`,
`test_algorithm.py::test_unused_select_clients_is_refused`.

### Added (step 1–2) and how it is checked

- `ts.data.partition` (iid / dirichlet / shards / quantity). `tests/test_data_partition.py`:
  every split is a seeded partition of the samples. Dirichlet's per-class share variance
  matches (1/K)(1−1/K)/(Kα+1) for α ∈ {0.1, 1, 10}, and quantity skew's size variance
  matches (1/K)(1−1/K)/(Kβ+1); both within 20 %. (My first test used (α+1), which is wrong:
  the code matched the correct formula, 0.0446 vs 0.045 at α = 0.1.)
- Datasets (mnist, fashion_mnist, cifar10/100, synthetic, npz) as in-memory tensors,
  normalized per channel; measured mean ≈ 0, std ≈ 1 on MNIST and CIFAR-10.
- `data:` config section; `ts.run(model_fn)` without `data_fn` builds the train loaders,
  the client split (saved as `client_split.json`) and the test set.
- Evaluation (`eval_data`, `data.eval_every`; local executor). Checked against an unsplit
  model evaluated by hand (loss at 1e-6, accuracy exact, a smaller last batch). Training
  losses are identical with and without evaluation, also with BatchNorm + Dropout and in
  parallel sessions. Sessions report per-client results.
- **Found while testing evaluation:** iterating any `DataLoader`, even unshuffled, draws one
  value from the global torch RNG (its worker seed). So evaluating between epochs shifted
  every later dropout mask: losses differed from epoch 2 (1.353257 vs 1.356807). Evaluation
  now runs inside `torch.random.fork_rng`. Any user callback that iterates a loader in
  `on_epoch_end` perturbs training the same way; the tutorial doesn't cover callbacks
  (open: a note in `monitor/callback.py`).
- Repo hygiene: `.gitignore` and `.git/info/exclude` had an unanchored `data/` that also
  ignored the new `torchslicer/data` package. Both anchored to `/data/`.

Verification of steps 1–2: tests 303/307 in the full run while another tenant loaded the GPU.
The 4 failures are GPU-only tests that step 2 doesn't touch: 3 × `test_offload.py::test_cuda_matches_local`
(the known NaN flake) and `test_transport.py::test_ipc_matches_tcp_on_gpu[hooks1]`. Rerun on an idle
GPU: 33/33 for both files, and the IPC test 6/6 in 3 more runs. So the IPC test is also load-sensitive
(open, like the offload one). CPU matrix 115/117 (the 2 known GPipe + BatchNorm cells); every
cell's losses identical to before M11.

### Step 3: many clients (2026-10-18)

Context: the container has 16 GB RAM, and exceeding it may get the container shut down. So
every scaling run had an estimate first and a watchdog (`benchmarks/clients/bench_clients.py`
kills itself above 11 GB of cgroup memory). The container peak stayed under 6 GB.

**Measured before changing anything** (old code):
- 1000 clients with a tiny client slice (the example CNN's conv1, 896 params): 2.0 ms/client,
  1.2 GB. Many clients were never the problem for small client slices.
- Per-client RAM ≈ 1.4 × (weights + optimizer state): 200 × 2.1M params used 4.7 GB vs
  3.4 GB of states. The extra came from transient deep copies.
- A switch deep-copied each state twice on top of the copies in `get_state`/`set_state`:
  4.4 ms (CPU) to 52 ms (GPU slices, CPU store) at 2.1M params; 233 / 478 ms at 33M.

**Changes:**
- `get_state` makes the only copy, `set_state` never aliases (`Optimizer.load_state_dict`
  keeps tensors already on the parameter's device, and Adam's `step` is updated in place),
  so `ClientSession` no longer deep-copies. Same-harness A/B: 2.1–3.4× faster on CPU,
  1.5–2.6× with GPU slices and a CPU store. Test: `test_stored_states_are_never_aliased_by_the_live_slices`
  (Adam, copy handoff). With a naive non-copying `set_state` it fails.
- `sl.client_states: cpu | gpu | cuda:i | auto`: where states live. GPU store switch:
  0.34–1.0 ms for 0.13M–33M params. Losses are equal to the CPU store at 1e-6
  (`test_gpu_client_store_matches_cpu_store`).
- Memory check before training (`resolve_client_store`): N × estimated state bytes must fit
  in 80 % of the store's free memory, else a `ValueError` with the numbers. Verified live:
  1000 × 2.1M params on the CPU store was refused (15.6 GB needed, 14.8 GB free) before any
  allocation.
- Client sampling (`Algorithm.select_clients`), in local and distributed sessions.
  `on_round_end`/`aggregate` see the participants; the aggregate goes to every client.
  SplitFed with 2-of-3 rotating sampling matches the plain-PyTorch reference at 1e-6.
  The `select_clients` refusal from issue 29 is lifted.

**Result** (SplitFed V2, 2.1M-param client slices, quiet GPU): 200 clients 9.2 → 3.7 ms/client
(CPU store) / 1.4 ms (GPU store); 1000 clients, impossible before (17 GB of states in 16 GB
RAM), now 1.5 s/round with 19.6 GB GPU and 1.0 GB RAM. The first measurement of the 200-client
rounds gave 27–47 ms/client while another tenant used the GPU; repeated on a quiet GPU it gave
the numbers above. GPU timings here are only meaningful when `nvidia-smi` shows it idle.

Verification of step 3: tests 314/314 on the last full run. The two runs before it each had
`test_transport.py::test_ipc_matches_tcp_on_gpu` failing at the launched subprocess's exit
code (not a loss mismatch). It passes alone (8/8) and after `test_sl.py`, so it is intermittent
in full runs, 2 of the last 4. The error text was lost to `tail`; open: capture it (the test
prints the subprocess's stdout/stderr on failure). CPU matrix 115/117 (known GPipe + BatchNorm),
all losses identical. Container memory peak during all of this: 6.2 GB.

### Step 4: heterogeneous clients, link-model calibration, more data (2026-10-18)

- **Client profiles** (`sl.client_profiles`, local executor; distributed refuses). Each
  client draws `{compute_scale, bandwidth_mbps, latency_ms}` by `share` (seeded).
  `CommMeter` keeps per-client sub-meters, so a client's simulated time is its own compute ×
  scale plus its own bytes and messages over its own link. Checked exactly: with scale 0, the
  time equals the link formula on the known bytes (2176 B and 3 messages per batch). The round
  equals the slowest client (parallel) or the sum (sequential) plus server compute, and
  training losses are unchanged.
- **Link-model calibration** (`benchmarks/transport/fit_link.py`): fit
  `other = 3·latency + 2·cut/bandwidth` per transport to the benchmark. A100 (cuts ≤ 10 MB):
  tcp 1.37 ms + 15.6 Gbit/s (median error 14 %, max 127 %), ipc 0.64 ms + 47 Gbit/s (7 %, 14 %),
  grpc 1.37 ms + 2.5 Gbit/s (11 %, 46 %). gRPC above 32 MB doesn't fit a line (issue 26).
  CPU (`results/cpu.json`, new): tcp 0.94 ms + 24 Gbit/s (12 %, 31 %), grpc 0.79 ms +
  3.1 Gbit/s (16 %, 45 %).
- **Simulator vs real**, CPU (the GPU was held at 99 % by another tenant throughout). Local
  executor with the fitted link vs the real 2-process run, ms/batch:
  medium/xlarge/act8 (cut ≥ 2 MB or compute-heavy): **0–6 % error** for both transports.
  small (~7 ms/batch): 42–58 % too low; large tcp: 33 % too low. The simulator misses fixed
  per-batch costs (RPC scheduling, thread handoffs) that dominate a few-ms batch, and two real
  processes share 4 CPUs, so their measured compute differs. Conclusion: simulated time is
  trustworthy when compute or cut size dominates, and optimistic for tiny models.
  **GPU** (idle A100, fits from `results/cuda.json`, 15 model × transport cases): median error
  12 %; cuts ≥ 3 MB 1–12 %; ipc 6–20 %; worst small/tcp 51 %, large/tcp 34 %, medium/tcp
  32 %. The simulator under-predicts everywhere but one case, as on CPU. (A first attempt
  ran under 99 % GPU load and used a fit file overwritten by a 100 MB fit, which gave gRPC a
  negative latency; both fixed: per-limit fit files, and runs only on an idle GPU.)
- **`natural` partitioner** (whole groups per client, greedy balance), **feature shift**
  (`noise`: std grows with the client index, fixed per sample; `rotate`), **`shakespeare`**
  (tiny-shakespeare by speaking role: 11,203 train windows, 241 roles; not LEAF's corpus),
  `npz` with `groups`. **Per-token accuracy** for `(N, C, T)` outputs, checked against the
  model evaluated by hand. Smoke: 10 clients by role, SplitFed: token accuracy 31 → 38 % in 2
  small epochs.

Verification of step 4: tests 320/325. The 5 failures are `test_offload.py` CUDA tests
(NaN), with the GPU held at 99 % by another tenant. Same-GPU A/B: 4 failures in each of 2
runs with step 4 and in each of 2 runs without it (one run gave 2), so they are not from this
change. But under load they now fail every time, including
`test_cuda_trains_a_model_that_does_not_fit`, which points at a real race in the offload
pager's stream/event ordering (M5), not noise. **Open, worth fixing before M16's offload
claims.** On an idle GPU, `test_offload.py` passes 25/25 three times in a row, so the
failures depend on load: a timing race, not wrong math.

## After M11 (2026-10-18, branch `fix/post-m11`)

**30. Offload trained to NaN under GPU load (M5, silent wrong result).** Symptom: the offload
CUDA tests failed with NaN losses whenever another process loaded the GPU, and passed when it
was idle. Reproduced on demand with a load generator (8192² matmuls in a loop, GPU at 100 %):
4 of 25 tests failed every run. Bisected by syncing at one place at a time (4 runs each):
- syncing after `acquire`/`fetch` or before `release`/`_writeback` didn't help; syncing before
  `_page_in`, or `CUDA_LAUNCH_BLOCKING=1`, fixed it;
- syncing the compute stream fixed it; syncing the H2D or D2H stream didn't;
- only full page-ins needed it (not state-only), and only the first copy into a **freshly
  allocated slot**: at construction or when slots grow after batch 1 (both needed).

So the first H2D into a new slot raced with work already queued on the compute stream, which
its `free`/`out` events (None for a new slot) don't cover. Which compute op touches the new
memory isn't identified (`_ensure_cap` synchronizes before allocating).
Fix: that one copy also waits for an event recorded on the compute stream (`_Slot.fresh`).
A first version waited at every page-in. That was also correct, but bimodally ~11 % slower
(0.168 ± 0.011 vs 0.159 ± 0.001 s/step, 150M-param model, 4 runs each). The narrowed version
measured 0.159 in 4 of 5 rounds, the same as before; in the fifth both versions slowed together
(external). After: `test_offload.py` 25/25 three times in a row at 100 % GPU load.

**31. The deploy code didn't reserve or expose the ports of ipc and nccl, and picked ports in
the ephemeral range (M3/M9).** Found by CI on `main` (py3.12): `test_nccl_path_on_gloo_matches_tcp`
failed with `EADDRINUSE` on the NCCL store port (driver port + 10), which the launcher never
checked. Also:
- `ipc` opens its tensor listener at port + offset like `tcp`, but the processes launcher,
  `ROLE=all`, k8s (Service/containerPort) and bare metal only reserved or exposed it for `tcp`.
  So `ipc` across k8s pods, which falls back to TCP on that port, would not connect;
- the NCCL store port wasn't exposed on the k8s lead Service;
- `free_port` took ports from the OS (32768–60999, the ephemeral range), so an outgoing
  connection could take one between the check and the bind.
Fix: `TransportConfig.listen_offsets(drives=)` is the one list of a node's ports (gRPC,
tensor listener for tcp/ipc, NCCL store on the driving node), used by every launcher;
`NCCL_STORE_PORT_OFFSET` is shared by the store and the workers; `free_port(offsets)` draws
from 20000–32000 and checks every offset. `start_worker(port=0)` refuses ipc like tcp.
Tests: `test_listen_offsets_per_transport`, `test_free_port_avoids_the_ephemeral_range_and_checks_every_offset`,
`test_render_k8s_exposes_ipc_and_nccl_ports`, `test_render_baremetal_spaces_ipc_nodes_like_tcp`.

Verification (issues 30–31 + GPU CI files): tests 330/330 with CUDA visible (offload, IPC
and the CUDA tests all passing); CPU matrix 115/117 (known GPipe + BatchNorm), all losses
identical.

## M12 privacy suite (2026-10-18, branch `feat/m12-privacy`, version 0.12.0)

### Added and how it is checked
- `torchslicer/privacy/`: `Attack` (observes one cut on the receiving side, after the cut
  hooks; scored against the batch's true inputs/labels, which the executor passes for scoring
  only), `Defence` (adds a client-side gradient term at a cut), metrics (AUC, distance
  correlation, PSNR, SSIM), registry + config `privacy.{attacks,defences}`, `train(privacy=)`.
  Built-ins: `NormLabelAttack`, `DirectionLabelAttack` (Li et al. 2022), `InversionAttack`
  (black-box inverse network trained online on aux queries), `DistanceCorrelation`, `NoPeek`.
- Hook point: `LocalExecutor._cut_forward/_cut_backward`, so standard, GPipe, local-loss,
  sessions and offload all get it. Tests `tests/test_privacy.py` (16):
  - observing never changes training: losses bit-identical with all attacks (standard and
    GPipe) and with the inversion attack (its decoder, sampling and seeding run in
    `torch.random.fork_rng`; `_query` restores each slice's mode and saved cut input `sl.x`);
  - `NoPeek(alpha=0)` bit-exact; `NoPeek(0.7)` equals a hand-written loop training the client
    on CE + α·dcor(x, z) (all parameters within 1e-6);
  - AUC equals its pairwise definition with ties; dcor = 1 on itself and on an orthogonal
    affine map, < 0.3 on independent data, finite gradient; SSIM matches its closed form on
    uniform images + noise;
  - qualitative: label AUC > 0.95 without defence, < 0.7 with gradient noise; NoPeek lowers dcor;
  - offload gives the same label AUC as local; refusals (distributed, offload + query attack,
    unknown names, bad cuts, duplicate keys, non-class labels).
- Tutorial `docs/tutorials/privacy.md` (executed by `test_docs.py`); benchmark
  `benchmarks/privacy/` (results in experiments.md, 2026-10-18).

### Design decisions and deviations from notes/privacy.md
- **NoPeek is a `Defence`, not an `Algorithm`** (the plan said Algorithm). As an Algorithm it
  could not be combined with SplitFed or local loss (one algorithm per run). Since
  d(α·dcor)/dz depends only on x and z, the client adds it to the returned gradient, which is
  exact (checked against the reference loop) and needs no graph change. Under GPipe it is
  per micro-batch, scaled 1/M like the task loss.
- **Distributed executors refuse privacy** (as planned): a remote observer has no ground truth.
  The check is the first thing in `DistributedExecutor.setup` (no discovery wait).
- Attack wall time is included in `duration_s`, not in `compute_s` (it runs outside the
  per-slice clocks).
- Bugs found in existing code: none.
- Matrix after M12 (CPU): 115/117, the 2 known GPipe + BatchNorm cells; losses unchanged.

### M12 step 4 (2026-10-18, same branch)
- Added: `FSHA` (malicious server), `UnSplit` (architecture only), `EmbeddingRecovery` (white-box
  LLM hidden states, Dong et al. ER), `DPAccountant` + the `ClipNoise` cut hook (per-sample clip,
  then Gaussian noise; GDP accounting). Access flags on `Attack`: `needs_client` →
  `client(fresh=False)` (a frozen `_ClientCopy` of slices 0..cut), `forges` → `forge(obs)`
  replaces the gradient at the start of `_cut_backward` (before the cut hook encodes it, so its
  bytes are counted and gradient noise still applies), scaled by the micro-batch scale.
- Core change: `SplitLayer.forward` split into the cut bookkeeping and `SplitLayer.run(x)`, the
  computation alone (no detached `self.x`), so a client copy passes gradients to its input
  across several slices. `forward` behaviour unchanged (tests + matrix).
- Tests (+12, `tests/test_privacy.py`): client-copy attacks leave losses bit-identical; the copy
  is frozen, in eval mode, passes input gradients, `fresh` re-initializes; a zero forged gradient
  freezes the client slice (SGD) and the server still trains, standard and GPipe; a forged
  per-micro-batch-mean gradient gives the same client weights with and without GPipe (M=4,
  1e-6); FSHA forge=False is bit-identical, forge=True changes training; UnSplit reconstructs
  smooth 8×8 images below 0.7× the zero-image MSE; ER recovers > 95 % of tokens on a tiny LM;
  GDP δ(ε) equals the hockey-stick divergence of N(μ,1) vs N(0,1) by numerical integration
  (rel 1e-6) and ε is below the classical σ-bound; `ClipNoise` clips rows to C and noises
  with std zC; the accountant reports `gaussian_epsilon(z, epochs, δ)` and refuses a missing,
  unclipped or wrong-direction hook; config names; offload refuses `needs_client`.
- Design decisions:
  - **FSHA default pilot ends in a ReLU when the smashed data are non-negative.** The paper's
    pilot is linear, but its client ends in a residual block; ours ends in ReLU+pool, so a linear
    target space cannot be matched. The server sees the smashed data, so this is within its
    knowledge. Measured: MNIST cut2, 1 seed, 2.8k iterations, SSIM 0.49 → 0.82.
  - **DP accounting scope**: per-sample, the sample's own row, ≤ 1 release per epoch,
    replace-one sensitivity 2C. Not end-to-end (weights trained on the sample influence later
    rows; BatchNorm on the sender mixes rows). Stated in the docstring and the tutorial.
  - **UnSplit TV is a per-pixel mean** (the paper sums); λ is then size-independent.
  - ER implements the paper's λ = 0 default (no distribution-matching penalty) and not TBS.
- Benchmark runner: `--seed k` / `--merge` (parallel seeds), per-run `attacks`, `dataset_params`
  (local npz), GPT-2 + Shakespeare.
- Observation: on the A100, passive-attack runs differ from the no-attack run at ~1e-4 in loss
  (cuDNN nondeterminism, the same between two plain runs); bit-exactness is asserted on CPU.
- Note: `load_dataset("cifar10")` downloads through torchvision even when `data/cifar10_*.npz`
  exist (the download ran at ~50 KB/s here); used `dataset: npz` instead. Not a bug, but a
  usability gap: worth a fallback to the local npz files.
- Bugs found in existing code: none.
- Matrix after M12 step 4 (CPU): 115/117, the same 2 GPipe + BatchNorm cells; losses unchanged.

## M13 baselines + Flower interop (2026-10-18, branch `feat/m13-baselines`, from `feat/m12-privacy`)

### Bugs found while reading the baseline papers
**32. Wrong SplitFed citation (docs).** `SplitFedAlgorithm`/`SplitFedV1Algorithm` cited
  arXiv:2108.09027, which is a physics paper ("Spin-to-Charge conversion with electrode
  confinement in diamond"). SplitFed is arXiv:2004.12088 (Thapa et al., AAAI 2022). Fixed.
**33. FedAvg was unweighted (silent deviation from the paper).** `fedavg` took the plain mean of the client states;
  SplitFed's Algorithm 1 (and FedAvg) weight client k by n_k / n. With unequal clients (any
  Dirichlet or quantity split) the aggregated model differed from the paper's. Fix:
  `fedavg(states, weights=None)`; `algorithm_round_end` passes the participants' sample
  counts as `metrics["client_samples"]` (`ClientSession.n_samples(k)`: the loader's dataset
  size, else its batches); both SplitFed algorithms take `weighting="samples"` (default,
  the paper) or `"equal"` (the old behaviour). Equal weights keep the old code path, so
  equal-size splits are bit-identical. Tests: weighted and equal SplitFed V1/V2 against a
  plain-PyTorch reference with clients of 1, 2 and 4 batches (1e-6); `fedavg` weights.

### Baseline template and reproductions
- `baselines/` (shipped): one config per paper with a `baseline:` section, a shared runner, `--check` against
  `expected {value, tol}`; CI runs every baseline for 1 epoch on synthetic data (`tests/test_baselines.py`).
- Runner bug found while running: a deep merge carried SGD's `momentum` into Adam's params (`_ATOMIC` keys
  are now replaced, not merged); seed files of `--runs` groups overwrote each other (suffix per group).
- Usability gap: `load_dataset("cifar10")` downloads through torchvision even when npz copies exist (50 KB/s
  here); baselines use `dataset: npz`.
- Reproductions: label leakage yes; SLPerf no (README lists the code differences, incl. a SLFrame vanilla
  optimizer-rebinding bug found by reading). Flower FedAvg = our weighted SplitFed within 1e-6 (checks #33).

### Gaps found by the SplitLoRA (GPT-2 + LoRA, E2E) baseline
**34. Frozen tied weights refused in multi-client sessions (over-strict check).** GPT-2 ties `wte` (client)
  and `lm_head` (server); `check_session_ties` refused any tie across a client and a server slice, so split
  LoRA fine-tuning with clients was impossible. A frozen tie never changes, so all clients see the same value.
  Fix: `TiedGroup.requires_grad`; frozen groups are exempt. Test: trainable tie still refused, frozen allowed
  and unchanged after training.
**35. No learning-rate schedule.** Every LLM fine-tuning recipe uses warmup + decay. New `training.scheduler`
  (`core/schedule.py`, `linear` = HF's linear warmup/decay, `register_schedule`): each slice writes
  lr = base x factor(step) right before its step, so client-state swaps can't leave a stale lr. Local executor
  only (offload and distributed refuse). Test: identical weights to plain PyTorch + LambdaLR.
**36. No user datasets in `data:` configs.** `ts.data.register_dataset(name, fn)` (like `register_partitioner`).
- Baseline runner: `extras.py` may define `evaluate(model, cfg, run, smoke)` (here beam search + the official E2E
  scorer) and register model builders; CI `synthetic` may override data/model/cuts.
- Overhead measured: GPT-2 S + LoRA, batch 8, A100: plain PyTorch 0.136 s/step, TorchSlicer (1 cut) 0.138.

- **M13 (C3-SL baseline):** new built-in hook `ts.CircularConv(ratio, seed)` (C3-SL batch-wise
  compression). Encode/decode are linear and mutual adjoints, so `encode_grad = bind`,
  `decode_grad = unbind` gives slice k exactly the autograd gradient of `decode(encode(z))` and
  compresses the backward by R too. Matches the authors' conv2d-based utils within 2e-5
  (fp32 vs our fp64 FFT); test `test_circular_conv_compresses_both_ways_and_gives_the_exact_gradient`.
  CIFAR-10 torchvision download ran at ~50 KB/s here: the baseline reads the local npz instead.

- **37. `fedavg` of one client kept its optimizer state** (`algorithm/builtin.py`): `n == 1`
  returned a deep copy of the state, optimizer included, while `n > 1` drops the optimizer
  ("re-warms on the next batch"). A one-client SplitFed run kept momentum across rounds,
  unlike every other client count. Found because Flower's engine (fresh optimizer per round)
  disagreed with `ts.SplitFedAlgorithm` on one client with momentum SGD. Fixed: `n == 1` drops it
  too; `test_fedavg_single` checks it.
- **M13 Flower engine:** `ts.SplitServer`/`ts.SplitClient` (`sl/remote.py`): the server slices
  over gRPC (generic handler, packed tensors, no proto, no pickle), one shared copy stepped per
  batch under a lock; `interop.flower.split_client_app` + `initial_client_arrays`. One remote
  client = local vanilla SL (Adam, 3 slices, with and without `CircularConv`); Flower
  `run_simulation`, 1 node = `ts.SplitFedAlgorithm` (momentum SGD), 3 nodes train. These tests run
  only where `flwr[simulation]` is installed (scratch venv here; skipped in the repo venv).

- **CI on the M13 merge PR (#5), 2026-10-20.** (a) py3.10: `flwr>=1.40` (the `flower` extra and the CI's
  `flwr[simulation]`) needs Python 3.11+, so the install failed. Fix: environment marker
  `python_version >= '3.11'` on both; on 3.10 the Flower tests skip. (b) py3.11/3.12: the SplitLoRA
  baseline's CI test needs `transformers`, which CI doesn't install. Fix: optional `baseline.requires`
  (packages), the test `importorskip`s each. Also `merge_to_main.sh` now stops on a missing git identity or
  a failed non-conflict merge (it used to swallow it with `|| true`).

## M10 prep: first real NCCL, across two containers (2026-10-20, branch `feat/m10-multi-client`)

Setup: this container (A100 `GPU-00b7e1d4`, bus 21:00.0, `10.224.0.11`) and a second container on
the same Proxmox host with the other A100 (`GPU-38553ba7`, bus 81:00.0, `10.224.0.9`), reached
over ssh. Different containers = different NCCL host hashes, so NCCL uses `NET/Socket` over eth0
(no P2P/SHM between them). Plain `torch.distributed` send/recv: bit-exact, ~1.27 GB/s at 32–64 MB
(virtual bridge, not a real network). Harness: `run_matrix.py` gained `MATRIX_REMOTE` (odd
workers / P2P followers over ssh), `MATRIX_PARTS` (2 slices = 2 GPUs) and a SIGUSR1 stack dump in
the matrix workers. Until today every NCCL check had run on gloo (CPU); these two bugs were
invisible there.

- **38. Real NCCL never initialized: `device_id` without an index** (`executors/worker.py`,
  since M9). Symptom: `NCCL rendezvous failed: slice 0: Please use a device_id with index.`
  Root cause: `resolve_device()` returns `cuda`, and `init_process_group(device_id=...)` (eager
  communicator) refuses an unindexed device in torch 2.11. Fix: `_indexed(device)` (`cuda` →
  `cuda:<current>`). Cost: none. Test: the cross-container matrix (`central-nccl`, `p2p-nccl`).
- **39. Real NCCL deadlocked at the first transfer** (`_send_forward_peer`, `_send_backward`,
  since M9). Symptom: after "NCCL group ready" both workers idle until the 240 s timeout; the
  stack dump shows rank 0 blocked *inside* `dist.isend`. Root cause: the sender called `isend`
  and only then the gRPC call carrying the shape, on the assumption that `isend` doesn't block
  ("kick off async NCCL send before gRPC metadata"). NCCL connects a pair of ranks lazily at
  their first transfer and `isend` blocks until the peer joins; the peer posts `dist.recv` only
  when the gRPC call arrives, which came after `isend`. gloo's `isend` doesn't block, so the
  CPU matrix passed. Fix: `_nccl_send` starts the metadata call first on a small thread pool
  (context copied, so trace spans stay attached; the routed/tunnel stubs have no `.future()`),
  then `isend().wait()`, then waits for the call. Same order guarantees as before. Test: the
  cross-container matrix.
- **Cross-container matrix after the fixes** (`benchmarks/equivalence/results/out_gpu_sgd_2hosts.txt`):
  9 models × {central, p2p} × {grpc, tcp, nccl}, CUDA, SGD, 2 slices (slice 0 here, slice 1 on the
  other GPU): **54/54 OK**. CNNs/TinyGPT(-LoRA) bit-exact; GPT-2/Llama(+LoRA) ≤ 4.8e-7 (HF attention
  kernels, as in the one-host CUDA matrix). First run of real NCCL, and the first cross-host run
  of every distributed mode. Harness problems on the way (not TorchSlicer): a port base inside
  the ephemeral range (`MATRIX_PORT_BASE` must stay < 32768), a log line glued to the JSON result
  (now parsed with `raw_decode`), remote workers surviving a killed harness (now killed at exit).
- **Transport benchmark across the two containers** (`bench_transport.py` with `BENCH_REMOTE`,
  `results/cuda_2hosts.json`, mean ± std of 3, ms/batch, P2P 2 slices, one GPU per container):

  | cut | grpc | tcp (pinned) | tcp-nopin | ipc (→ TCP) | nccl (socket) |
  |---|---|---|---|---|---|
  | 0.07 MB (small) | 4.15 ± 0.50 | 3.31 ± 0.04 | 2.92 ± 0.16 | 2.94 ± 0.02 | 3.56 ± 0.22 |
  | 2.1 MB (medium) | 19.8 ± 1.6 | 10.4 ± 1.4 | 8.5 ± 0.4 | 11.3 ± 0.8 | 9.7 ± 0.2 |
  | 8.4 MB (act8) | 63.9 ± 0.4 | 19.1 ± 2.5 | 16.8 ± 1.0 | 15.4 ± 0.5 | 17.7 ± 0.9 |
  | 33.6 MB (act32) | 660 ± 47 | 60.8 ± 4.7 | 101.7 ± 2.7 | 63.0 ± 4.6 | 81.6 ± 8.7 |
  | 67.1 MB (act64) | 1313 ± 26 | 111.4 ± 7.7 | 189.2 ± 5.4 | 110.3 ± 0.2 | 136.1 ± 11.4 |

  Reading: (1) **NCCL is not faster than pinned TCP here**: ~equal up to 8 MB, 0.75–0.82× at
  32–64 MB. Expected: between containers NCCL uses its *socket* transport over the same 10 Gbit
  virtual bridge (no RDMA, no NVLink, no P2P), so it moves the same bytes through the same
  kernel sockets, with its own chunking on top. Its advantage needs GPUDirect RDMA / NVLink, not
  available on this host. Default NCCL socket settings (`NCCL_SOCKET_NTHREADS`,
  `NCCL_NSOCKS_PERTHREAD`) not tuned. (2) Pinned TCP is at the link bound: act64 moves 2 × 67 MB
  per batch, 106 ms at the 1.27 GB/s measured with plain NCCL send/recv, vs 97.6 ms of non-compute
  time. (3) `ipc` across hosts falls back to TCP (same numbers), as designed. (4) Pinned buffers
  pay off from 32 MB (1.7×), as on one host. (5) gRPC still collapses above 32 MB (issue 26; the
  malloc env vars were not set here). Not a real network: same physical host, bridge.

## M10 multi-client engine, step 1–2: client nodes (2026-10-20, branch `feat/m10-multi-client`)

### Added and how it is checked
- `sl.client_nodes` (centralized): each client is a worker holding slice 0 and its own loader
  (`start_worker(data=, client_index=)`, `ts.run` `ROLE=client` + `CLIENT_INDEX`); the
  coordinator (`ts.ClientNodesExecutor`) holds no data and no slice; `ts.RemoteSession` is
  the schedule. Protocol: `client_info`, `client_step`, `get_state`, `set_state` RPCs;
  `SliceConfig.prev_workers` (slice 1: every client) and `label_worker` (a client: the last
  slice); `WorkerStatsResponse.bytes_label/msgs_label`. Per-batch return address = the batch
  id: `(batch_id // n_micro) % n_clients`, so no frame format changed.
- Equivalence (`tests/test_client_nodes.py`, 13 tests): sequential copy / keep / callable
  handoff, parallel, parallel + `SplitFedAlgorithm`, client sampling, GPipe (2 micro-batches),
  over gRPC and TCP: losses equal the in-process `ClientSession` (to its 6-decimal rounding)
  and every client's final weights within 1e-6. Same through `ts.run` (`ROLE=all`) and with
  `launch --target processes` (7 OS processes: 2 server workers, 4 clients, coordinator) on
  `experiments/sl_client_nodes.yaml`. `bytes_state` = 0 where no state needs to move (keep,
  plain parallel, sampling only), > 0 with a handoff copy or aggregation.
- No bug found in existing code. Behaviour kept: every existing executor path unchanged
  (two base hooks, `_batch_id` and `_round_end`, default to the old code; the gradient send now
  picks its peer per batch with `_prev_for`, which returns the single previous worker when
  there are no client nodes).
- Refused, with a reason: `nccl` (fixed process group), `u_shaped` (later), fault tolerance
  (step 4), per-client server copies (step 3), k8s rendering (identical pod template).
- **Across the two containers** (2026-10-20): `launch --target baremetal` on
  `experiments/sl_client_nodes.yaml`, coordinator + 2 server workers on `10.224.0.11`, the 4
  client nodes on `10.224.0.9` (registration with the coordinator over the bridge): losses
  `[2.369131, 2.207663]`, equal to the in-process `ClientSession`; every node exited cleanly.
  First run of client nodes on two hosts (same physical machine: correctness only).

### Step 3: SplitFed V1 on client nodes (2026-10-20)
- Per-client server copies live on the server workers (`SliceConfig.n_clients`,
  `per_client_state`): `WorkerServicer._select_client` swaps the batch's client's copy in at
  the forward entry (`_forward`, `_forward_tensor`, on the compute pool), keeping the previous
  client's; a client's first batch of a round starts from the slice as the round began
  (= `LocalExecutor`'s `initial_server_state` per epoch). Round end: the coordinator reads
  each participant's copy from every server worker (`get_state(per_client, client)`), calls
  `algorithm_round_end(..., server_states)`, sends the aggregate and `reset_clients`.
- Checked: `SplitFedV1Algorithm` parallel (TCP), sequential keep, with client sampling, and
  with GPipe: losses and final client weights = in-process V1 (V1 and V2 losses differ here,
  `[1.392442, 1.387596]` vs `[1.400225, 1.367958]`, so the copies are really exercised).
- Gap found on the way (no bug): `SlicedModel.train` passes `setup` only the arguments its
  signature names (`inspect.signature`), so an executor overriding `setup` with `*args,
  **kwargs` would silently lose `planner` and `cut_hooks`. `ClientNodesExecutor.setup`
  repeats the base signature (checked equal).
- Cost: each round moves every participant's server copy to the coordinator and the
  aggregate back (`bytes_state`); a server-side aggregation would avoid it (later).

### Step 4a: client nodes that die; round deadline (2026-10-20)
- `sl.drop_failed_clients` (default true): a client node that dies leaves the run. Death =
  a failed call or heartbeat **and** a failed liveness probe (`_gone`), so a live client that
  refuses a batch (a bug) still stops the run. Its batch in flight is abandoned (new base hook
  `_batch_lost`, default False: not counted in the loss), its later batches are skipped, no
  state moves to or from it, and it leaves the aggregation. Sequential `copy` across a lost
  client hands on what that client started from (its own updates are lost with it). A dead
  *server* worker still fails the run (`WorkerFailureError`), unchanged. History:
  `clients_lost`, `batches_skipped`. Client error reports all name slice 0, so the
  coordinator confirms one by probing every live client for a fatal error.
- `sl.round_deadline_s`: no batch is scheduled after that many seconds in a round.
- Batch ids now come from a per-run counter (`_seq * n_clients + client`), so an abandoned
  batch's id is never reused.
- Checked: client 1 killed at its batch 1 (parallel) / batch 0 (sequential copy) of round 1
  = an in-process session where it had only the batches it finished and then left
  (losses and the survivors' weights); with SplitFed the survivors share the aggregate; with
  `drop_failed_clients: false` the run stops; a 0.5 s deadline with 0.2 s client steps skips
  part of every round. Regression found on the way (in this step's own code, before commit):
  `keep` fetched the previous client's state for nothing.
- Cost / limits: a batch abandoned mid-way may already have updated the server slices (the
  server steps before it sends the gradient back); this is inherent to losing the client.

### Step 4b: clients that join late or come back (2026-10-20)
- `sl.min_clients` (0 = all): the run starts with the server workers and at least that many
  client nodes. The coordinator scans the discovery's `known_nodes()` (new on
  `CoordinatorDiscovery` and `StaticDiscovery`; a node is up if `describe` answers in 1 s).
  At each round's start (`_admit`), a missing or lost client that answers gets slice 0
  (`_slice0()`), starts from `RemoteSession.broadcast` (the last state given to every client,
  e.g. a SplitFed aggregate) or else the initial weights, and slice 1 learns its address
  (new RPC `set_peers`; `""` = no client there). History: `clients_joined`.
- **Bug found (in this step, before commit): a restarted client was admitted only half the
  time.** Symptom: 2 of 4 identical runs left client 1 out of round 2 (`clients_lost` again,
  loss 1.357295 instead of 1.375115). Root cause: the coordinator's `Router` caches one gRPC
  channel per address; the channel to the dead client was in reconnect backoff, so the 1 s
  readiness probe expired before it redialed, although the new process was up. Fix:
  `Router.reset(address)` drops the cached channel (and its call objects); `_admit` resets
  every candidate address before probing. Cost: one new channel per absent client per round.
  After: 6/6 runs identical, and the late-join / rejoin tests pass 3/3 repeats.
- Checked: a client absent in round 1 joins round 2 = an in-process twin that leaves it out
  of round 1 (with and without SplitFed: it gets the aggregate, as non-participants do
  in-process); a client killed in round 1 and restarted at the same address rejoins round 2
  from the initial weights = a twin that resets it after round 1 (losses and every client's
  weights). A fixed peer list that can't meet the counts fails at once, not after the timeout.
