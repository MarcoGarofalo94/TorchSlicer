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
