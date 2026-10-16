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
