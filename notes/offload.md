# Out-of-core training: technology survey (M5, step 1)

Created 2026-10-15. Question: how to train a model whose weights + optimizer state + activations
don't fit in GPU memory (and, as a second tier, not in RAM) on **one node**, by paging slices in
from a slower tier. Literature rules apply: only what was retrieved, with a link; anything not
read in the source is marked `[unverified]`.

Legend for tiers: G = GPU, C = CPU DRAM, N = NVMe/disk. "Offloaded" = what lives in the slow tier.

## Systems and papers

### vDNN (Rhu et al., MICRO 2016)
- https://arxiv.org/abs/1602.08124
- Tiers G/C. Offloaded: layer input feature maps (activations) only; weights stay on the GPU.
- Overlap: 2 CUDA streams (compute, memcpy); pinned host buffers + `cudaMemcpyAsync`; offload
  during the same layer's forward; in backward, prefetch the input of an earlier layer while the
  current one computes; a pre-reserved memory pool avoids synchronous `cudaMalloc/cudaFree`.
- Numbers: vDNN_dyn ~97% of baseline throughput on average; worst case VGG-16 (batch 256) on a
  12 GB Titan X, 18% loss vs. an oracle GPU with enough memory.
- License/callable: research system on cuDNN, not a library. Ideas only.

### L2L, layer-to-layer (Pudipeddi et al., arXiv 2020)
- https://arxiv.org/abs/2002.05645
- Tiers G/C. Offloaded: the whole model + optimizer state + fp32 master weights live on the host
  ("eager param-server", EPS); the device holds one layer + transit buffers + the stashed
  boundary activation of each micro-batch.
- Schedule: forward layer by layer, each layer over **all** micro-batches before the next one is
  loaded; backward in reverse, **recomputing** each layer's forward from its stashed input; the
  optimizer runs on the host.
- Overlap: the timing model adds transfer to compute (no overlap assumed); overlapping
  reduction/update with execution is left as future work ("L2Lp").
- Numbers: BERT-Large, 4 V100: inner micro-batch loop 52.5 → 84.9 samples/s for 1 → 4
  micro-batches (+60%); claims 50B params on one 16 GB V100 with 512 GB host RAM.
- **Closest to the M5 plan** (slices = layers, boundary stash + recompute, layer-major micro-batches).
- Code: not found `[unverified]`.

### SwapAdvisor (Huang, Jin, Li, ASPLOS 2020)
- https://par.nsf.gov/biblio/10191573
- Tiers G/C. Joint search (genetic algorithm) over operator schedule, memory allocation and swap
  decisions on the dataflow graph (MXNet).
- Numbers: models up to 12x the GPU memory, 53–99% of an infinite-memory baseline's throughput.
- Ideas only (MXNet).

### ZeRO-Offload (Ren et al., arXiv 2101.06840; USENIX ATC 2021 `[unverified venue]`)
- https://arxiv.org/abs/2101.06840 (read via https://ar5iv.labs.arxiv.org/html/2101.06840)
- Tiers G/C. Offloaded: fp32 master params, Adam momentum/variance, fp16 gradients, and the
  optimizer computation (CPU-Adam). fp16 params and fwd/bwd stay on the GPU. Activations are not
  offloaded (activation checkpointing instead).
- Overlap: gradients copied to CPU per parameter/bucket as soon as computed, overlapping the rest
  of backward; updated params copied back tile by tile while CPU-Adam works on the next tile.
  Optional one-step **delayed parameter update** (changes the math; convergence "slightly slower
  early on").
- Numbers: CPU-Adam 5–6.4x faster than PyTorch CPU Adam (1B params: 0.22 s vs 1.39 s; GPU Adam
  0.10 s); 13B params on one 32 GB V100; 40 TFLOPS on a 10B model; CPU↔GPU volume 4M bytes per
  step (M = params) vs 28M for L2L.
- License: Apache-2.0 (DeepSpeed, https://github.com/deepspeedai/DeepSpeed). Callable only through
  the DeepSpeed engine (wraps the whole model + optimizer).

### ZeRO-Infinity (Rajbhandari et al., arXiv 2104.07857; SC21 `[unverified venue]`)
- https://arxiv.org/abs/2104.07857 (read via https://ar5iv.labs.arxiv.org/html/2104.07857)
- Tiers G/C/N. Offloaded: params, grads, optimizer state to C or N; activation checkpoints to C.
- Overlap: a dynamic prefetcher traces the operator sequence and overlaps N→C, C→G and G↔G
  transfers with compute; DeepNVMe (async bulk I/O, near-peak sequential bandwidth); a pinned
  buffer manager reuses "tens of GB" of pinned buffers for "tens of TB" of state;
  memory-centric tiling of large operators.
- Numbers: >25 PFLOPS on 512 V100 (40% of peak); >40 TFLOPS/GPU single node up to 100B;
  activation offload >80% efficiency at hidden ≥ 8K with ~3 GB/s per GPU.
- License: Apache-2.0 (DeepSpeed). Same engine constraint as ZeRO-Offload.

### DeepNVMe (DeepSpeed ops, ≥ 0.15.0)
- https://www.deepspeed.ai/tutorials/deepnvme/
- Async file ↔ tensor I/O (`aio_handle`: `async_pread/async_pwrite/wait`, needs `libaio`;
  `gds_handle` for GPUDirect Storage), pinned-buffer helpers. Examples use it standalone
  (`AsyncIOBuilder().load().aio_handle()`) without the training engine.
- Numbers (their hardware): 3.69 GB/s read / 3.18 GB/s write on a 5.6/4.3 GB/s rated SSD after tuning.
- **Callable from TorchSlicer** as an optional disk-tier backend.

### PatrickStar (Fang et al., arXiv 2108.05818)
- https://arxiv.org/abs/2108.05818
- Tiers G/C. Model data managed in chunks placed dynamically between CPU and GPU, using memory
  statistics from a warm-up iteration. Claims 2.27–2.5x the model scale of DeepSpeed.
- Engine-level (wraps the model), not callable per slice.

### Harmony (Li et al., VLDB 2022)
- https://arxiv.org/abs/2202.01306 (design read from https://arxiv.org/html/2202.01306v2),
  code https://github.com/msr-fiddle/harmony (MIT, built on PyTorch 1.5, 5 commits)
- Tiers G/C (+ GPU p2p). Swaps weights, gradients, intermediate tensors; recomputes inside a
  "pack" (contiguous layers) from the pack's checkpointed input.
- Schedule: **input-batch grouping** = run one pack over all micro-batches back to back (one weight
  swap instead of m); just-in-time weight update as soon as a pack's grads are ready; balanced-time
  packing. Disabling grouping: 2.2x slower (DP); disabling prefetch: up to 1.3x slower.
- Overlap: 5 CUDA streams per GPU (compute, swap-in, swap-out, p2p-in, p2p-out) + CUDA events;
  background CPU threads for pinned allocation, prefetch, waits; double-buffered prefetch; one
  memory manager pooling memory across streams.
- Numbers: up to 7.6x over virtualized-memory baselines; up to 2.3–3.2x over ZeRO-Infinity;
  per-minibatch loss matches the baseline (BERT-Large/MRPC). No fraction-of-PCIe-bound reported.
- Model must be decomposed into per-layer code by its tools; old PyTorch. Ideas, not a dependency.

### STRONGHOLD (Sun et al., SC22)
- https://sc22.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap381.html
- Tiers G/C/N. Dynamically offloads to CPU RAM and secondary storage, keeping the minimum working
  window on the GPU; 1.9–6.5x model size and 1.2–3.7x throughput vs prior offloading on a 32 GB V100.
- Code/license not checked `[unverified]`.

### FlexGen (Sheng et al., ICML 2023)
- https://arxiv.org/abs/2303.06865 (design read from https://ar5iv.labs.arxiv.org/html/2303.06865),
  code https://github.com/FMInference/FlexGen (Apache-2.0)
- Tiers G/C/N. **Inference only.** Offloads weights (layer granularity), activations and KV cache
  (tensor granularity), with an LP that picks the percentages per tier.
- Schedule: column-by-column "zig-zag" block schedule = one layer over all batches of a block
  (weight I/O amortized, within 2x of optimal I/O); per step, in parallel: load next layer's
  weights, store previous batch's activations/cache, load next batch's, compute current.
- Numbers: OPT-175B at 1 token/s on one 16 GB GPU (effective batch 144).
- Same amortization idea as L2L/Harmony, for inference; not usable for training.

### LoHan / Fuyou → Ratel (Liao et al.; arXiv 2403.06504; ICDE 2025)
- https://arxiv.org/abs/2403.06504 ; Ratel: https://doi.org/10.1109/ICDE65448.2025.00029
  (same authors; arXiv v1 "Fuyou", v2 "LoHan", ICDE version "Ratel")
- Tiers G/C/N. Fine-tuning 100B+ on one consumer GPU with limited RAM; co-optimizes SSD↔CPU
  traffic with compute; active gradient offloading; traffic-aware activation swapping.
- Numbers: 175B on an RTX 4090 with 256 GB RAM; 156 vs 45 TFLOPS against ZeRO-Infinity on 13B
  (v1); up to 2.32x over SOTA on 13B (v2). Design internals not read `[unverified]`.

### SSDTrain (Wu et al., arXiv 2408.10013; DAC 2025)
- https://arxiv.org/abs/2408.10013
- Tier N for **activations** only, transfers fully overlapped with compute; −47% activation peak
  memory with negligible throughput loss. Mechanism details not read `[unverified]`.

### HF Accelerate big-model `device_map` / disk offload
- https://huggingface.co/docs/accelerate/usage_guides/big_modeling
- Tiers G/C/N. **Inference**: per-module hooks move each layer's weights to the GPU when an input
  arrives and remove them afterwards; "only one GPU will be active at any given moment". No
  training support or prefetch described on the page. Apache-2.0. Not a training baseline.

### PyTorch built-ins (installed: torch 2.11)
- **FSDP2 `fully_shard(offload_policy=CPUOffloadPolicy(pin_memory=True))`** — source docstring in
  `torch/distributed/fsdp/_fully_shard/_fsdp_api.py`: params, grads and optimizer state on CPU;
  params copied H2D before all-gather, grads D2H in backward, **optimizer step on CPU**; pinned by
  default "for the copies to overlap with compute". Needs a process group (world size 1 works).
  FSDP1 docs (https://docs.pytorch.org/docs/2.14/fsdp.html): `CPUOffload(offload_params=True)`
  also offloads grads, so the step runs on CPU; no grad accumulation outside `no_sync()`.
  BSD; zero install → **usable as a baseline**.
- **`torch.autograd.graph.save_on_cpu(pin_memory=True)`** — saved-tensor hooks: activations saved
  for backward go to (pinned) CPU and come back asynchronously on unpack. Activations only.
- **`torch.utils.checkpoint`** (https://docs.pytorch.org/docs/2.14/checkpoint.html) — recompute;
  `preserve_rng_state=True` stashes CPU + one device type's RNG state and restores it at recompute,
  so dropout masks match; the docs warn that a function behaving differently at recompute (global
  state) gives wrong gradients. BatchNorm running stats are not discussed: a module that updates
  buffers in forward updates them again at recompute — we must handle that ourselves.

## What the survey says about the design
1. The winning schedule for "model bigger than GPU" is the L2L/Harmony one, the same as FlexGen's
   for inference: page a block (slice) in, run it over **all micro-batches**, stash only boundary
   inputs, recompute in backward, update the block's weights **just in time** right after its
   backward. This is exactly TorchSlicer's slice + `SplitLayer.x` + per-slice optimizer, so M5 is
   a scheduler over existing slices, not a new engine.
2. Overlap needs separate copy streams + events, pinned host buffers allocated once and reused,
   double-buffered device slots, and prefetch of the next block (vDNN, Harmony, ZeRO-Infinity).
3. Where the optimizer runs is a real trade-off: ZeRO-Offload/FSDP-offload run it on the CPU
   (moves params down + grads up = less PCIe traffic, but CPU Adam is 5–14x slower than GPU Adam
   per their numbers); L2L runs it on the host too; Harmony says it can be either. We measure it.
4. Disk tier: mature async I/O exists (DeepNVMe); plain buffered reads into pinned memory on a
   background thread are the zero-dependency fallback.

## Decision (2026-10-15)
- **Write** (in TorchSlicer, ~one module): an offload path for the local executor,
  `pipeline.offload: none|cpu|disk`. Each slice's parameters (+ optimizer state when the step runs
  on the GPU) have a flat home buffer in pinned host RAM (or a file); two device slots of the
  largest slice size are allocated once; a slice is paged in by one `non_blocking` copy on a
  dedicated H2D stream and its parameters are re-pointed at views of the slot (no allocator
  churn), paged out on a D2H stream. Schedule = L2L/Harmony: forward slice by slice without a
  graph over all GPipe micro-batches, keeping only boundary inputs; backward in reverse order with
  recompute from the boundary input, backward, optimizer step, page out, while slice i−1 is being
  prefetched. Cut hooks and the `CommMeter` stay in the loop (`_cut_forward/_cut_backward`).
  Reason: no surveyed tool can be called per slice of an arbitrary fx-sliced model — DeepSpeed,
  PatrickStar and FSDP wrap the whole model and own the optimizer; Accelerate and FlexGen are
  inference-only; Harmony needs its own per-layer model rewrite on PyTorch 1.5.
- **Reuse** from PyTorch: pinned memory, `torch.cuda.Stream`/`Event`, `torch.random.fork_rng`
  (the same RNG stash/restore as `checkpoint(preserve_rng_state=True)`), and from TorchSlicer:
  `SplitLayer`, the local executor's cut/metrics code, `balanced_cuts` memory caps (extended with
  an activation + optimizer-state estimate), the profiler and the equivalence harness.
  **Optional extra:** DeepNVMe `aio_handle` for the disk tier if `deepspeed` is importable;
  default disk backend = background-thread reads into pinned buffers.
- **Don't reuse** `torch.utils.checkpoint` itself: the boundary is the slice, whose input is
  already detached (`SplitLayer.x`), so recompute is "run the slice again with the saved RNG
  state"; wrapping it in `checkpoint` would add nothing and keep its graph alive across slices.
  BatchNorm-style buffers are snapshotted before the recompute and restored after, so running
  stats update exactly once.
- **Optimizer placement:** `pipeline.offload_optimizer: gpu|cpu|auto`; `auto` picks by a measured
  micro-benchmark (GPU step + state transfer vs CPU step).
- **Baseline for speed (same model, batch, GPU cap):** FSDP2 + `CPUOffloadPolicy` at world size 1
  (built into the installed torch, zero install). DeepSpeed ZeRO-Offload/Infinity as a second
  baseline if it installs here (`pip download` works; its CPU-Adam op is JIT-compiled) `[to check]`.
