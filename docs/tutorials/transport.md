# Tensor Transport

How activations and gradients travel between slices: `transport.tensor` in the
`RunConfig` YAML (or the `TENSOR_TRANSPORT` env var). The data owner's inputs/labels
and all control calls stay on gRPC whatever the mode.

| Mode   | Use it for | Peers |
|--------|------------|-------|
| `grpc` | anything; NAT tunnels and relays (default) | any |
| `tcp`  | directly reachable peers | any; gRPC through tunnels |
| `ipc`  | GPU slices on the same machine | same machine; TCP otherwise |
| `nccl` | one GPU per slice, multi-GPU or RDMA clusters | any, one GPU each |

Measured on one A100, 2 slices in 2 processes, steady state, mean ± std of 3
(`benchmarks/transport/`):

| cut per batch | grpc | tcp | ipc |
|---|---|---|---|
| 0.07 MB | 6.8 ms | 3.4 ms | 3.7 ms |
| 2.1 MB | 24.2 ms | 12.4 ms | 5.0 ms |
| 8.4 MB | 61.2 ms | 15.3 ms | 7.9 ms |
| 67 MB | 1411 ms | 95.4 ms | 20.8 ms |

Below ~1 MB the transport hardly matters; above it, `tcp` beats `grpc` and `ipc`
beats both.

**CPU threads.** torch sizes its thread pool from the host's cores, not a container's
CPU quota. Oversubscribed, the CPU work on the receive path spins idle threads and
burns the quota: gRPC/TCP ran 4–5× slower on a 4-CPU container. Set `OMP_NUM_THREADS`
to the quota on every node.

**Cuts of 32 MB or more.** glibc serves each buffer that large with fresh memory, and
the first write to every page of it costs a page fault (~0.6 ms/MB here). gRPC makes about
nine copies per transfer, so it suffered most: 724 ms/batch at 33.6 MB vs 61 ms at 8.4 MB.
Setting these on every node restores linear scaling (gRPC 654 -> 175 ms at 33.6 MB,
1276 -> 348 ms at 67 MB):

```bash
export MALLOC_MMAP_MAX_=0 MALLOC_ARENA_MAX=1 MALLOC_TRIM_THRESHOLD_=4294967296
```

The cost: the process keeps its peak memory, and one malloc arena serializes
allocations across threads. Prefer `tcp`/`ipc` for large cuts where you can.

## tcp

A raw socket per directly reachable peer. CUDA tensors are copied into a reused
pinned host buffer (`TCP_NO_PINNED=1` disables it). That matters above 32 MB, where
a fresh pageable buffer page-faults on every send: 1.8× faster at 32–64 MB, no
difference below 8 MB.

## ipc

The sender exports the GPU buffer as a CUDA IPC handle (~350 bytes) over the TCP
socket. The receiver maps it, clones it on the GPU and ACKs, so the sender can
release it. No host copy: a 67 MB activation moves in ~8 ms vs ~81 ms on TCP. CPU
tensors and peers on another machine use TCP. Cut hooks work: hooked payloads are
packed on the GPU (`pack_payload`), so they go through IPC too.

## nccl

`torch.distributed.isend`/`recv` on a process group (rank = slice index). The
coordinator hosts the rendezvous `TCPStore` at its port + 10. gRPC carries only the
shape/dtype, and the tensor travels on NCCL. Cut hooks work (packed GPU payloads).
- **One GPU per slice.** NCCL refuses two ranks on one GPU. TorchSlicer checks this
  at init and stops with the reason; on one GPU, use `ipc`.
- Every rank reports its rendezvous to the coordinator, which stops the run as soon
  as one fails, instead of hanging at the first transfer.
- On CPU it runs on gloo, the same code path (the tests use it).
- Not yet run on multi-GPU hardware: treat it as experimental.

## Benchmark

```bash
.venv/bin/python benchmarks/transport/bench_transport.py                 # all
.venv/bin/python benchmarks/transport/bench_transport.py act8 act64 --variants tcp,ipc
```
