# Tensor Transport

TorchSlicer supports four tensor-transport modes, set via `transport.tensor` in your
`RunConfig` YAML (or `TENSOR_TRANSPORT` env var).

## Modes

| Mode   | When to use | Same-node | Cross-node |
|--------|-------------|-----------|------------|
| `grpc` | default; works everywhere | ✓ | ✓ |
| `tcp`  | directly reachable peers; 1.5–2× over gRPC | ✓ | ✓ |
| `ipc`  | same-machine GPU workers; zero-copy D2H | ✓ | falls back to TCP |
| `nccl` | GPU-to-GPU P2P / GPUDirect RDMA; best throughput | ✓ | ✓ (GPUDirect) |

## grpc (default)

Tensors are serialised into protobuf `Tensor` messages. Works across machines,
through NAT tunnels, and with all TorchSlicer features including cut hooks.

## tcp — pinned-memory async D2H

Raw TCP socket between directly reachable peers. A pre-allocated pinned host
buffer is reused per connection; `tensor.to(pinned, non_blocking=True)` + a CUDA
event lets the D2H copy overlap with header construction, synchronising only when
`sendall` needs the bytes. Falls back to gRPC for peers behind NAT or relay.

```yaml
transport:
  tensor: tcp
```

## ipc — CUDA IPC zero-copy

For same-machine, same-CUDA-context workers. The sender exports the GPU buffer as
a CUDA IPC handle (~350 bytes) sent over a TCP socket; the receiver reconstructs the
tensor without any H2D copy. An ACK byte ensures the handle is consumed before the
sender frees the buffer. Automatically falls back to TCP for CPU tensors or
cross-machine peers.

```yaml
transport:
  tensor: ipc
```

**Limitation**: requires CUDA and same-machine workers. Incompatible with NAT tunnels
(the TCP socket must be directly reachable). Not compatible with `cut_hooks`.

## nccl — NCCL P2P / GPUDirect RDMA

Uses `torch.distributed.isend` / `irecv` over a NCCL ProcessGroup (rank = slice
index). The coordinator hosts a `TCPStore` master for rendezvous; workers connect
at `coordinator_port + 10`. gRPC carries only shape/dtype metadata (empty `data`
field = NCCL sentinel); the tensor travels on the NCCL channel.

```yaml
transport:
  tensor: nccl
```

**Limitations**:
- All workers must share compatible CUDA and NCCL installations.
- Incompatible with `cut_hooks` (tensors travel raw on NCCL).
- Persistent workers re-initialise the ProcessGroup on each run; the previous group
  is destroyed first (`dist.destroy_process_group`).

## Benchmark

```bash
# all four modes, all model sizes, 3 seeds each
.venv/bin/python benchmarks/transport/bench_transport.py

# specific models only
.venv/bin/python benchmarks/transport/bench_transport.py small large

# specific transports
.venv/bin/python benchmarks/transport/bench_transport.py --transports grpc,tcp,ipc
```
