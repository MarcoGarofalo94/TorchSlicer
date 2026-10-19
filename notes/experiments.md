# Experiments log (append-only)

Each entry: date, hypothesis, config, command, key metrics, 1-line conclusion.

---

## 2026-10-16 — M9 transport benchmark: TCP pinned-memory vs gRPC baseline

**Hypothesis**: Pinned-memory async D2H in TCP transport reduces ms/batch vs gRPC by 1.5–2×
for medium-to-large activations on GPU.

**Branch**: `feat/m9-fast-transport`

**Config**: inline (no YAML) — P2P, vanilla setting, 2 slices, batch=64, 4 epochs (warm-up=1),
REPEATS=3 seeds, DEVICE=cuda (A100 MIG 3g.40gb).

**Command**:
```
.venv/bin/python benchmarks/transport/bench_transport.py --transports grpc,tcp
```

**Results** (mean ± std ms/batch, 3 seeds):

| Model   | Act. MB/cut | gRPC ms/batch | TCP ms/batch | Speedup |
|---------|-------------|---------------|--------------|---------|
| small   |  1.0 MB     |  9.96 ± 9.43  |  5.42 ± 0.07 | 1.84×   |
| medium  | 33.6 MB     | 29.46 ± 1.18  | 13.46 ± 0.81 | 2.19×   |
| large   |  8.4 MB     | 14.24 ± 1.16  | 10.23 ± 0.25 | 1.39×   |
| xlarge  | 50.3 MB     | 39.74 ± 2.05  | 30.90 ± 1.88 | 1.29×   |

Note: "small" gRPC std is high (±9.4 ms) due to first-batch JIT + CUDA context init jitter on
very fast layers (total ~10 ms); the TCP std is tight (±0.07), confirming the pinned path is
stable. Medium sees the best speedup (2.19×) as expected: large enough activation to pay for
gRPC serialisation, small enough that the D2H is not the bottleneck.

**Conclusion (gRPC vs pinned TCP)**: TCP delivers 1.3–2.2× speedup over gRPC (the serialisation
overhead dominates). The pinned-memory improvement itself is a separate question — see below.

---

## 2026-10-16 — M9 transport benchmark: TCP pinned vs TCP sync (old path)

**Hypothesis**: Async pinned-memory D2H (`non_blocking=True` + CUDA event) is faster than
synchronous `.cpu().contiguous()` for the TCP send path on GPU.

**Method**: `TCP_NO_PINNED=1` env-var forces `_send_cpu()` (the old synchronous path) even on
CUDA tensors, same benchmark harness.

**Results** (mean ± std ms/batch, 3 seeds):

| Model | Old TCP (sync) | New TCP (pinned) | Δ |
|-------|---------------|-----------------|---|
| small  1 MB  | 4.73 ± 0.50 ms | 5.01 ± 0.27 ms | –6% |
| medium 34 MB | 13.24 ± 1.00 ms | 13.65 ± 1.69 ms | –3% |
| large  8 MB  | 10.27 ± 0.43 ms | 9.65 ± 0.17 ms | +6% |
| xlarge 50 MB | 30.29 ± 2.23 ms | 31.22 ± 0.59 ms | –3% |

**Conclusion**: No measurable improvement from pinned memory in this benchmark. All differences
are within run-to-run noise. Root cause: on loopback the socket is the bottleneck, not D2H.
The header construction that was meant to overlap with the DMA is only ~microseconds of CPU
work (a few struct.pack calls), so there is nothing real to hide behind. The `non_blocking` +
event machinery adds a small but visible bookkeeping overhead.

Pinned memory would help when: (a) D2H is genuinely slow (slow PCIe, many slices competing
for bandwidth), (b) there is substantial CPU work between kick-off and sendall to fill, or
(c) multiple slices send concurrently and the DMA engine is being exercised in parallel.
None of these apply in the current single-machine loopback benchmark.

The architecture is correct and worth keeping — it adds negligible overhead on loopback and
provides headroom for multi-node NVLink/PCIe-saturated scenarios — but the speedup claim must
be qualified: **no speedup on loopback/local; expected benefit on bandwidth-limited PCIe buses**.

**IPC / NCCL**: not benchmarked in the thread-based harness — requires separate processes.
See the multi-process benchmark below.

---

## 2026-10-16 — Multi-process transport benchmark: gRPC vs TCP vs IPC (subprocess workers)

**Hypothesis**: CUDA IPC zero-copy delivers significant speedup over gRPC and TCP for
same-machine GPU→GPU tensor transfers when workers run as separate OS processes.

**Branch**: `feat/m9-fast-transport`

**Config**: inline — P2P vanilla setting, 2 slices. Follower worker launched as a real
subprocess (`subprocess.Popen`) — separate CUDA context, real IPC. REPEATS=3, EPOCHS=4
(warm-up=1), DEVICE=cuda (A100 MIG 3g.40gb).

**Command**:
```
.venv/bin/python benchmarks/transport/bench_multiproc.py medium large xlarge --transports grpc,tcp,ipc
```

NCCL excluded: hangs on MIG single-instance (both processes land on the same GPU; NCCL's
communicator probes NVLink/PCIe P2P between what it sees as the same device and deadlocks).
NCCL requires genuinely separate GPU devices or multi-node.

**Results** (mean ± std ms/batch, 3 seeds):

| Model   | Act. MB/cut | gRPC ms/batch      | TCP ms/batch      | IPC ms/batch      | IPC vs gRPC |
|---------|-------------|-------------------|-------------------|-------------------|-------------|
| medium  | 33.6 MB     | 103.91 ± 35.32    | 74.75 ± 1.88      | 23.41 ± 0.35      | **4.44×**   |
| large   |  8.4 MB     |  70.23 ±  0.56    | 68.38 ± 1.12      | 23.16 ± 0.26      | **3.03×**   |
| xlarge  | 50.3 MB     |  90.17 ±  7.21    | 90.86 ± 2.07      | 38.30 ± 2.60      | **2.35×**   |

**Observations**:

- **IPC delivers 2.3–4.4× over gRPC** across all tested sizes. The zero-copy GPU→GPU handle
  (no serialisation, no D2H, no H2D) is the dominant factor.
- **TCP is barely faster than gRPC** in the subprocess benchmark (1.03–1.39×). The subprocess
  overhead and loopback socket bottleneck cancel the serialisation savings. The pinned-memory
  improvement remains undetectable (same conclusion as the thread-based benchmark).
- **IPC improvement decreases as activation size grows**: medium (34 MB) gets 4.4× but xlarge
  (50 MB) gets 2.35×. The likely cause: at 50 MB the IPC ACK round-trip cost becomes
  non-trivial, and the gRPC numbers are dominated by serialisation + socket time, so both
  paths are slower but IPC scales better.
- **gRPC in the subprocess benchmark is ~3–4× slower** than in the thread-based benchmark
  (103 ms vs 29 ms for medium). The subprocess adds real socket latency, IPC context switching,
  and gRPC channel negotiation overhead that is elided by in-process stub calls.

**Conclusion**: CUDA IPC is the right transport for same-machine multi-GPU split learning.
Implement on production hardware (two discrete GPUs or two nodes) to see the full benefit.
TCP/gRPC parity on loopback confirms that TCP's advantage only appears at the network level.

---

## 2026-10-17 — M9 transport benchmark, redone (corrects the three 2026-10-16 entries above)

**Why redone**: the 2026-10-16 numbers have three errors (details: validation.md M9
"Corrections"):
1. "Act. MB/cut" was per epoch, not per batch: real cuts were 0.07–3.15 MB.
2. ms/batch included setup (discovery, probing, init).
3. Threads were oversubscribed: 32 torch threads on a 4-CPU quota inflated gRPC/TCP 4–5×.

Their conclusions (IPC "2.3–4.4×", pinned "no benefit", NCCL "hangs on MIG") are superseded.

**Hypotheses**:
- (a) pinned D2H helps once cuts are large;
- (b) IPC's transfer is ~10× faster than TCP's at large cuts;
- (c) the transport matters little for sub-MB cuts.

**Branch**: `feat/m9-fast-transport`. **Config**: in `benchmarks/transport/bench_transport.py`:
- P2P vanilla, 2 slices; the follower is a subprocess (own CUDA context);
- SGD(0.01, 0.9), 16 batches, 4 epochs (epoch 1 = warm-up, excluded);
- seeds 0–2, `OMP_NUM_THREADS` = cgroup quota (4), 1× A100 80GB PCIe (seen as 1 device).

`act{8,32,64}` = MLP whose cut is 8/32/64 MB per batch.

**Command**: `.venv/bin/python benchmarks/transport/bench_transport.py` (~15 min) →
`benchmarks/transport/results/cuda.json`.

**Results** (steady-state ms/batch, mean ± std of 3; "other" = ms/batch − measured compute):

| model | cut MB/batch | grpc | tcp | tcp no-pin | ipc | ipc other vs tcp other |
|---|---|---|---|---|---|---|
| small  | 0.07 | 6.77 ± 3.78 | 3.44 ± 0.04 | 3.52 ± 0.09 | 3.73 ± 0.04 | 2.1 vs 1.8 ms |
| large  | 0.52 | 10.38 ± 0.03 | 9.49 ± 0.70 | 9.81 ± 0.06 | 4.18 ± 0.04 | 2.0 vs 4.7 |
| medium | 2.10 | 24.18 ± 0.75 | 12.40 ± 0.83 | 11.92 ± 0.66 | 5.03 ± 0.06 | 2.3 vs 8.3 |
| xlarge | 3.15 | 32.43 ± 3.59 | 17.24 ± 0.24 | 17.74 ± 0.16 | 11.88 ± 0.28 | 3.4 vs 8.6 |
| act8   | 8.39 | 61.24 ± 3.47 | 15.33 ± 1.02 | 15.19 ± 0.40 | 7.86 ± 0.49 | 4.7 vs 11.8 |
| act32  | 33.55 | 724.19 ± 11.55 | 51.28 ± 3.33 | 92.49 ± 8.21 | 12.99 ± 1.28 | 5.5 vs 42.9 |
| act64  | 67.11 | 1411.48 ± 34.72 | 95.37 ± 4.12 | 168.16 ± 5.60 | 20.78 ± 0.02 | 7.8 vs 81.3 |

NCCL: skipped. It needs one GPU per slice, and this host exposes one.

**Observations**:
- (a) **Confirmed above 32 MB.** Pinned is 1.8× over the old path at 32 MB and 64 MB, and
  makes no difference up to 8 MB. Cause: the old path's `.cpu()` page-faults on fresh
  allocations above glibc's 32 MB mmap threshold (micro: 20.3 vs 1.4 ms at 33.6 MB).
- (b) **Confirmed.** IPC's non-compute time is 10.4× lower than TCP's at 64 MB (7.8 vs
  81.3 ms) and 7.8× lower at 32 MB. End to end: 4.6× over TCP and 68× over gRPC at 64 MB.
  Micro: a 33.6 MB IPC round trip takes 0.69 ms vs 24.6 ms for a loopback socket (1.3 GB/s).
- (c) **Confirmed.** At 0.07 MB, TCP ≈ IPC; IPC wins from ~0.5 MB.
- gRPC collapses above 8 MB: 61 → 724 ms for 4× the bytes. The cause isn't investigated; a
  plausible one is the same >32 MB fresh-allocation page faults on its several copies
  `[unverified]`.

**Conclusion**: for same-machine GPU slices use `ipc`; elsewhere use `tcp`. Keep `grpc` for
NAT/relay, and avoid it for cuts over a few MB. NCCL remains unmeasured.

## 2026-10-17 — gRPC transport above 32 MB

- Hypothesis: gRPC's superlinear slowdown (61 ms at 8.4 MB -> 724 ms at 33.6 MB) comes from
  page faults on fresh ≥32 MB buffers, not from gRPC itself.
- Command: `BENCH_REPEATS=2 .venv/bin/python benchmarks/transport/bench_transport.py act32
  act64 --variants grpc,tcp-nopin`, with and without `MALLOC_MMAP_MAX_=0 MALLOC_ARENA_MAX=1
  MALLOC_TRIM_THRESHOLD_=4294967296`; before/after the `deserialize_tensor` copy fix.
- grpc ms/batch at 33.6 / 67 MB: 740 / 1423 before; 654 / 1276 with the fix; 175 / 348 with
  the fix plus the malloc settings. tcp-nopin: 101 / 181 -> 49 / 97 with the malloc settings
  (= pinned TCP).
- Conclusion: confirmed. The code fix gives 10–12 %; the malloc settings give 4×. gRPC stays
  3.6× slower than TCP at these sizes. Details: notes/validation.md issue 26.

## 2026-10-18 — Does the simulator's link model predict real transport time?

- Hypothesis: `sim time = compute + messages·latency + bytes/bandwidth`, with latency and
  bandwidth fitted per transport, predicts real per-batch time of a 2-process P2P run.
- Commands: `CUDA_VISIBLE_DEVICES= BENCH_REPEATS=2 benchmarks/transport/bench_transport.py small
  large medium xlarge act8 --variants grpc,tcp`; `benchmarks/transport/fit_link.py results/cpu.json`;
  local executor with the fitted `sl.link_*` on the same models (scratch script).
- Result (CPU): error 0–6 % for medium, xlarge, act8 (both transports); 6–58 % for small and
  large, always under-predicting.
- GPU (idle A100, grpc/tcp/ipc): median error 12 %, cuts ≥ 3 MB 1–12 %, worst 51 % (small, tcp).
- Conclusion: confirmed when compute or cut size dominates; optimistic for few-ms batches
  (fixed per-batch overheads are not modelled).

## 2026-10-18 — M12 privacy suite: label leakage and input inversion (MNIST)

- Hypotheses: (1) Li et al. 2022: on an imbalanced binary task the returned cut gradients leak
  labels (leak AUC ≈ 1); isotropic gradient noise brings it to ≈ 0.5. (2) He et al. 2019: a
  black-box inverse network reconstructs inputs well at a shallow cut and worse deeper;
  activation noise and NoPeek lower it at an accuracy cost.
- Configs: `benchmarks/privacy/configs/{label,inversion}_mnist.yaml` (10k train / 2k test, CNN,
  Adam 1e-3, batch 64, 3 epochs, seeds 0–2, A100). Command:
  `.venv/bin/python benchmarks/privacy/run_privacy.py <config>` (smoke with `--smoke` first).
  Results: `benchmarks/privacy/results/*.json`. Total 2.3 min.
- Label ("is it a 0?", ~10 % positives, cut after conv1+pool), last epoch, mean ± std:
  no noise: norm AUC 0.86 ± 0.02 (q95 0.98), direction AUC 1.00 (q95 1.00), test acc 0.997.
  Grad noise std 1e-4: norm 0.55 ± 0.02, direction 0.52; 3e-3: 0.51 / 0.50; test acc unchanged
  (0.9965–0.9968). The q95 stays ~0.71–0.79 even with noise: per-batch AUC on ~6 positives is
  that noisy at chance, so q95 is a weak summary at this batch size.
- Inversion (PSNR dB / SSIM / dcor / test acc): cut1 (16×28×28) 42.6 / 0.996 / 0.999 / 0.974;
  cut4 (32×14×14) 26.4 / 0.946 / 0.985 / 0.975; cut8 (128) 15.4 / 0.459 / 0.846 / 0.974.
  cut1 + act noise 0.5: 30.5 / 0.976, acc 0.965; noise 2: 24.9 / 0.928, acc 0.927.
  NoPeek cut1 α=1: no change; α=10: 36.2 ± 4.8 dB, dcor 0.96, acc 0.968. cut4 α=10: 18.6 dB,
  dcor 0.64, acc 0.930; α=100: training collapses (acc 0.116, chance), dcor 0.004.
- Conclusion: both qualitative results reproduced. Direction attack exact as in the paper;
  the norm attack weakens once the model is confident (mean 0.86, q95 0.98). On this easy task
  gradient noise removes the leakage with no measurable accuracy cost, because the noise
  std is absolute and swamps tiny gradients, and the server slices alone reach 99.7 %:
  not a privacy/utility trade-off curve, a harder task is needed for that (negative result
  for the trade-off). NoPeek needs a deeper client (one conv layer can barely decorrelate)
  and has a sharp collapse between α = 10 and 100.

## 2026-10-18 — M12 step 4: malicious server, architecture-only and LLM inversion, label DP

- Hypotheses: (1) FSHA (Pasquini et al. 2021): a malicious server forging the returned
  gradient makes a non-queryable client invertible, beyond an honest server with the same
  networks. (2) UnSplit (Erdogan et al. 2022): architecture-only coordinate descent recovers
  inputs at shallow cuts. (3) ER (Dong et al. 2025): white-box inversion of GPT-2 hidden states
  recovers most tokens, less with depth. (4) Clipped Gaussian gradient noise (ClipNoise) gives
  a privacy/utility curve with an ε, on a task whose gradients do not vanish.
- Configs: `benchmarks/privacy/configs/{fsha_mnist,unsplit_mnist,llm_gpt2,dp_label_cifar}.yaml`,
  seeds 0–2, A100; `run_privacy.py <config> [--smoke | --seed k | --merge]` (smoke first; seeds
  as parallel processes). Memory < 10 GB.
- FSHA, MNIST CNN, 30k train, 3 epochs (1.4k setup iterations; paper ~1e3–1e4), last-epoch SSIM
  of the reconstruction / test acc: cut2: black-box inversion (queries) 0.981 ± 0.001 /
  0.985; honest server, same networks 0.157 ± 0.049 / 0.985; **FSHA 0.552 ± 0.269** / 0.974
  (seeds 0.73, 0.69, 0.24). cut5: inversion 0.893; honest 0.036; FSHA 0.150 ± 0.039, acc
  0.884 ± 0.038; FSHA + NoPeek α=10: 0.063, acc 0.62 ± 0.44 (one seed collapses).
- UnSplit, MNIST CNN, 2 epochs, 9 inverted batches of 8 per run, 30×(20+20) Adam steps:
  MSE ≈ the zero image's at every cut (0.97–1.03; zero image ≈ 1.0 on normalized MNIST),
  clone MSE 53 → 329 with depth. A 10k-step probe (cut1, lr 1e-3 and 1e-2): MSE 0.91 / 0.93.
- ER, GPT-2 (124M) fine-tuned on Shakespeare (800 × 32 tokens, AdamW 5e-5), 3 inverted
  batches of 4 sequences, 1000 AdamW steps lr 1e-2, token accuracy (F1 the same ±0.005):
  cut1 0.926 ± 0.014, cut3 0.836 ± 0.016, cut6 0.684 ± 0.024, cut9 0.579 ± 0.033. cut1 +
  activation noise std 0.1: 0.924 (test loss unchanged 3.886); std 1: 0.596 ± 0.016 (test loss
  3.945, +0.06).
- Label DP, CIFAR-10 "airplane?" (10 % positives), 20k train, 3 epochs, cut2, clip C = 1e-2
  (≈ p90 of the noise-free gradient row norms), δ = 1e-5. Norm AUC / direction AUC / test acc /
  ε: none 0.80 / 0.94 / 0.931; clip only 0.84 / 0.96 / 0.918; z 0.03: 0.85 / 0.89 / 0.913 /
  7159; z 0.1: 0.67 / 0.60 / 0.915 / 748; z 0.3: 0.52 / 0.51 / 0.923 / 115; z 1: 0.50 / 0.50 /
  0.925 / 20.1; z 3: 0.50 / 0.50 / 0.924 / 5.2. Unclipped std 1e-3 (= z 0.1's std): 0.80 /
  0.73 / 0.922.
- Conclusions: (1) partly reproduced: FSHA hijacks at the shallow cut (SSIM 0.55 vs 0.16 for
  the honest control, 2 of 3 seeds ≈ 0.7) but at our budget it is far below a black-box
  inverse network (0.98) and fails at the deeper cut, where it costs 10 points of accuracy
  (the paper's 1e4 iterations would be > 30 min here; not run). (2) **Negative**: UnSplit does
  not beat the zero image on this CNN and normalization at any cut or budget we tried; the
  paper's own MNIST figure (0.048 MSE, pixel scale not stated) is about half a zero image's
  error in [0, 1]. (3) Reproduced: ER recovers 93 % of tokens after one block and degrades
  with depth (58 % after 9 of 12), the paper's motivation for TBS on deep layers. (4) A
  real curve now: the attacks fall to chance at z ≥ 0.3 for ~1 point of accuracy (mostly the
  clipping), long before ε is meaningful (ε = 5.2 at z = 3): the heuristic attacks need far
  less noise than the worst-case bound. Clipping alone does not protect, and clipping makes
  the noise effective (z 0.1 vs unclipped noise of the same std: direction AUC 0.60 vs 0.73).

## 2026-10-18 — M13 baselines: SLPerf Table 2 (MNIST) and Li et al. label leakage
- Hypothesis: TorchSlicer reproduces published SL numbers from one config (`baselines/<name>/config.yaml`,
  `baselines/run.py <name> --seed k`, `--merge`, `--check`), 3 seeds, 9 processes in parallel.
- SLPerf (arXiv:2304.01502), LeNet cut after conv block 1, 3 clients, batch 64, 30 epochs, last-epoch test
  acc (ours vs paper): vanilla IID 0.9902 ± 0.0012 vs 0.9792; vanilla Dir0.1 0.8982 ± 0.0545 vs 0.7288;
  U-shape IID 0.9899 ± 0.0008 vs 0.9728; U-shape Dir0.1 0.9385 ± 0.0046 vs 0.7019; SplitFed IID
  0.9755 ± 0.0031 vs 0.9882 (best epoch 0.9853 ± 0.0012); SplitFed Dir0.1 0.8021 ± 0.0231 vs 0.9768.
  Non-IID split = SLFrame's `hetero` ported (`slperf_hetero`); a first pass with ts.data `dirichlet` (discarded)
  gave vanilla/U-shape Dir0.1 0.83–0.95, same picture.
- SLFrame code reading: vanilla `client_manager.py:67` rebinds `trainer.model = torch.load(...)` while the
  optimizer (client.py:23) holds the old parameters → client layers stop training after the first handoff
  [code read, not run]. Consistent with SLPerf's lower vanilla/U-shape. Eval: every client on the full test
  set after its turn (= ours). SplitFed Dir0.1 gap (0.80 vs 0.98) unexplained.
- Li et al. (arXiv:2102.08504), CIFAR-10 airplane binary stand-in, cut2, 3 epochs, q95 leak AUC: direction
  1.000 ± 0, norm 0.993 ± 0.009 (paper ≈ 1); ClipNoise C 1e-2 z 1: direction 0.746 ± 0.035, ε 20.1.
- Conclusion: label leakage reproduced; **SLPerf not reproduced** (IID within 0.011–0.017, outside ±0.01;
  non-IID ranking reversed). Not tuned towards the paper; a vanilla run that freezes the client after the
  first handoff would test the SLFrame-bug explanation (not run).
- 2026-10-18 follow-up: ran SLFrame itself (scratch venv + MPICH, my driver: torch/numpy shims, unused
  imports stubbed, polling sleeps capped; same accuracies with and without the caps). Vanilla Dir0.1 seed 0,
  30 epochs, last client: 0.946 published code, 0.959 with the optimizer rebuilt after each handoff
  (bug confirmed, +1.3 pt only). SLFrame is at 0.71–0.73 after 2 epochs → Table 2's 0.7288 is not SLFrame's
  30-epoch result [setting of Table 2 unverified]. U-shaped in SLFrame = private client head/tail per
  client (no handoff), never terminates (epoch counted twice, stop at == 3); 0.55–0.67 after 2–4 rounds ≈
  paper's 0.70. Conclusion: the vanilla/U-shape gaps are the paper's, not TorchSlicer's.
- 2026-10-19 SLFrame batches B+C (seed 0, 30 epochs, last client): vanilla IID 0.987; U IID 0.988 (mean of
  clients); U α0.1 0.738 (0.649/0.767/0.799); SplitFed IID 0.980; SplitFed α0.1 0.932 (0.60–0.94 across
  epochs). SLFrame ≈ TorchSlicer everywhere except SplitFed α0.1 (0.93 vs our 0.80 ± 0.02): open; next test =
  train TorchSlicer on SLFrame's exact seed-0 split. Batch C took 74 min (3 runs on 4 CPUs), not the 25 estimated.
- 2026-10-19 exact split: TorchSlicer on SLFrame's seed-0 α0.1 split (dumped from SLFrame, label order
  checked), 3 seeds: vanilla 0.959 ± 0.004 (= SLFrame with optimizer fix 0.959); SplitFed last 0.890 ± 0.018,
  best 0.925 (SLFrame 0.932 last, last-5 mean 0.876). Conclusion: the 0.80 vs 0.93 gap was the split;
  TorchSlicer = SLFrame; SLPerf Table 2 ≠ its own code. Script: scratchpad slperf_driver/ts_exact.py.

## 2026-10-19 — SplitLoRA (GPT-2 S, r = 4, E2E), seed 0, full recipe
- Hypothesis: TorchSlicer reproduces SplitLoRA Table I (CenLoRA BLEU 69.41, SplitLoRA 68.79), tol 0.01.
- Config `baselines/splitlora_e2e/config.yaml`; `baselines/run.py splitlora_e2e --seed 0 --runs <run>` (both in parallel on the A100, 2 h 14 min each).
- cen_lora: BLEU 0.5604, NIST 7.842, METEOR 0.401, ROUGE-L 0.622, CIDEr 1.429.
- split_lora: BLEU 0.5627, NIST 7.842, METEOR 0.409, ROUGE-L 0.628, CIDEr 1.449.
- Both ~13 BLEU below the paper; split ≈ centralized, as in the paper (gap there 0.6). Not reproduced.
- Found after the run: HF GPT-2 trains with attn/resid/embd dropout 0.1; the LoRA recipe's GPT-2 (examples/NLG/src/model.py) has none (only lora_dropout). A recipe-fidelity bug in the baseline, fixed (`gpt2_lora(pdrop=0.0)`). Predictions were not saved; now `E2E_PREDS` writes them.
- Conclusion: SL vs centralized parity holds; absolute BLEU gap open, rerun with dropout fixed pending the owner's OK.

## 2026-10-19 — SplitLoRA rerun, GPT-2 dropout 0 (LoRA recipe), seed 0
- Same command, `gpt2_lora(pdrop=0.0)`, predictions saved (E2E_PREDS).
- cen_lora: BLEU 0.5658, NIST 7.826, METEOR 0.402, ROUGE-L 0.628, CIDEr 1.462.
- split_lora: BLEU 0.5665, NIST 7.869, METEOR 0.417, ROUGE-L 0.631, CIDEr 1.425.
- Dropout was not the cause (+0.5 BLEU). Outputs are fluent and cover the slots; mean length 29.0 words vs refs 27.2. Name copies are often off (`Crown Plaza` for `Crowne`, which train refs also have 189 times; `The sorrento`).
- Eval path checked against microsoft/LoRA gpt2_decode.py (grouping by context, cut at <|endoftext|> / "\n\n", no lowercasing/tokenizing for E2E): same.
- Conclusion: not reproduced, ~12.5 BLEU below SplitLoRA Table I in both settings; SL vs centralized parity holds (0.567 vs 0.566). Next: run microsoft/LoRA's own code with gpt2.sm to see whether the recipe reaches ~69 at GPT2-S.
