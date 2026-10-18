# Training a model that doesn't fit: out-of-core slices

`ts.OffloadExecutor` trains a model whose weights, gradients and optimizer state don't fit in
GPU memory, on one node. Every slice lives in a slower tier (pinned host RAM, or files) and is
paged into the GPU when it computes, while the next slice is already being copied in. The
losses are the same as training the whole model in GPU memory: the
[equivalence matrix](../../benchmarks/equivalence/) checks every model at 1e-5.

The design follows L2L and Harmony:

1. **Forward**, slice by slice, without an autograd graph. Only each slice's input (the cut
   tensor) and the RNG state are kept.
2. The **last slice** runs forward with a graph, then the loss and backward.
3. **Backward** in reverse slice order. The slice's forward is recomputed from its stored
   input with the same RNG state, so dropout masks match. Then comes backward, then the
   optimizer step right away, and the slice is paged out. While this happens, the previous
   slice is already being paged in.

## The smallest example

```python
import torch
import torch.nn as nn
import torchslicer as ts

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                      nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))
data = [(torch.randn(16, 32), torch.randint(0, 4, (16,))) for _ in range(4)]

device = "cuda" if torch.cuda.is_available() else "cpu"
sliced = ts.slice(model, n=4, executor=ts.OffloadExecutor(mode="cpu", device=device))
history = sliced.train(data, {"name": "AdamW", "params": {"lr": 1e-3}},
                       {"name": "CrossEntropyLoss", "params": {}}, epochs=2)
print(history[-1]["loss"], history[-1]["offload_h2d_bytes"])
```

The model stays where you built it, on the CPU. After training, its parameters are ordinary
CPU tensors that hold the trained values, so `model.state_dict()` works as usual.

## Options

| `OffloadExecutor(...)` | config `offload:` | env | meaning |
|---|---|---|---|
| `mode` | `mode` | `OFFLOAD` | `cpu`: pinned host RAM; `disk`: one file per slice and region in `dir` |
| `optimizer` | `optimizer` | `OFFLOAD_OPTIMIZER` | where the optimizer step runs: `gpu` pages the optimizer state with the slice; `cpu` sends the gradients to the host and steps there, in a background thread; `auto` measures which is faster |
| `directory` | `dir` | `OFFLOAD_DIR` | disk mode only |
| `slots` | `slots` | `OFFLOAD_SLOTS` | device slots, at least 2: one slice computes while the next is copied in |
| `device` | `device` | `OFFLOAD_DEVICE` | compute device (default `network.device`) |
| `direct_io` | `direct_io` | `OFFLOAD_DIRECT_IO` | disk mode: `O_DIRECT` instead of the page cache |

With a `RunConfig`, setting `offload.mode` is enough. `sliced.train(..., run_config=cfg)` then
uses an `OffloadExecutor` instead of the default `LocalExecutor`:

```python
cfg = ts.RunConfig()
cfg.offload.mode = "cpu"
cfg.offload.device = device
cfg.logging.enabled = False
sliced = ts.slice(model, n=4)
sliced.train(data, {"name": "SGD", "params": {"lr": 0.01}},
             {"name": "CrossEntropyLoss", "params": {}}, epochs=1, run_config=cfg)
assert isinstance(sliced.executor, ts.OffloadExecutor)
```

GPipe micro-batches (`use_gpipe=True, n_micro_batches=M`) amortize each page-in over `M`
micro-batches: a slice runs all of them before the next slice is paged in. This is the
"input-batch grouping" from Harmony.

## Choosing the slices

Each slot must hold one slice: its parameters, buffers and gradients, plus the optimizer
state when the step runs on the GPU. The GPU must also hold the activations of one slice's
recompute and every slice's input. More slices make each slot smaller, but don't change the
bytes moved per step: every slice is paged in and out once per pass either way. Use the
`params` strategy to make the slices even:

```python
sliced = ts.slice(model, strategy="params", n=4, executor=ts.OffloadExecutor(device=device))
```

## What you get in the history

Every epoch entry has the usual SL metrics (bytes per cut, compute per slice), plus:

- `offload_h2d_bytes` and `offload_d2h_bytes`: bytes paged in and out during the epoch;
- `offload_home_mb` and `offload_slots_mb`: the slow-tier footprint and the device slots;
- on CUDA, `offload_peak_mb`: the peak allocated device memory;
- on CUDA, `offload_h2d_gbps` and `offload_d2h_gbps`: the measured pinned-copy bandwidth;
- on CUDA, `offload_transfer_bound_s` and `offload_bound_fraction`: the time those bytes need
  at that bandwidth, and its share of the epoch. A share near 1 means the run is
  transfer-bound and the copies run at full speed.

## Limits

- **Shared tensors stay resident.** A tensor used by several slices stays on the GPU and is
  not paged: tied embeddings, or the rotary buffer every Llama block shares. With
  `optimizer="cpu"`, a *trainable* shared parameter is refused.
- **Dropout with micro-batches.** With dropout in several slices and `M > 1` micro-batches,
  offload draws its dropout masks in a different order from the GPipe executor: it runs slice
  by slice, GPipe runs micro-batch by micro-batch. The masks have the same distribution, but
  the runs are not bit-identical. Neither matches an unsplit full-batch run. With
  `M = 1`, all three are identical.
- **Disk mode.** Slice files are read and written in 16 MiB requests over 4 I/O threads,
  through pinned staging buffers. By default this goes through the page cache, which the
  kernel reclaims as needed. `direct_io=True` (config `offload.direct_io`) uses `O_DIRECT`
  instead, so the bytes really go to the SSD. Measure both on your machine: where M5 was
  measured, `O_DIRECT` was 3x slower. Between pages, and after
  training, the parameters are memory-mapped views of the files. They cost no RAM until
  read, so `state_dict()` works even for a model larger than RAM.
- **The cpu tier must fit in RAM.** `mode="cpu"` keeps the parameters and the optimizer state
  in pinned host memory. Setup checks this against the available memory, including a
  container's cgroup limit, and refuses with the numbers rather than being OOM-killed later.
- **A CPU optimizer step needs a fast CPU.** With `optimizer="cpu"`, pass `fused: true` in the
  optimizer params (fused CPU AdamW is about 5x faster than the default). `auto` measures
  the CPU step against the transfers it saves and picks the faster placement.
- **Local topology only.** Distributed workers don't page their slices (yet).
