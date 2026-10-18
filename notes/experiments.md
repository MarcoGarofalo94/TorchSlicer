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
