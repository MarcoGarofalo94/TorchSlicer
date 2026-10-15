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
