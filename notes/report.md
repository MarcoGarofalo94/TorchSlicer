# TorchSlicer feature walkthrough (started 2026-10-26)

For each feature: what it is, a short code example, a proposed test, and the result of running it.
Scripts are in `notes/report/`, run with `.venv/bin/python notes/report/<script>.py`.
Status: ✅ checked · ⚠️ checked, issues found · ⏳ not reviewed yet.

## Feature inventory

| # | Feature | Milestone | Where | Status |
|---|---|---|---|---|
| F1 | Slicing: trace, strategies, named cuts, `expand=` | M0–M1 | `core/model_graph.py`, `strategies/` | ⚠️ (refinements on `refinement`) |
| F2 | `pack=` for models fx can't trace (HF LLMs, MoE), tied weights | M1 | `core/stages.py`, `core/tied.py`, `examples/pack/` | ⏳ |
| F3 | Local executor + GPipe micro-batching, bf16 | M0 | `executors/local.py`, `core/split_layer.py` | ⏳ |
| F4 | Centralized gRPC (coordinator + workers), TCP tensors | M0 | `executors/distributed.py`, `executors/worker.py` | ⏳ |
| F5 | P2P (worker 0 = driver), `ts.run_p2p` | M2 | `executors/p2p.py` | ⏳ |
| F6 | Capability-aware planning (`describe`, links, `balanced_cuts`) | M2 | `strategies/cost.py`, `distributed.py` | ⏳ |
| F7 | Fault tolerance: heartbeats, `WorkerFailureError`, recovery from checkpoint | M2 | `distributed.py`, `testing_faults.py` | ⏳ |
| F8 | Run anywhere: `ts.run`, `launch --target local/processes/compose/k8s/baremetal` | M3 | `deploy/` | ⏳ |
| F9 | SL settings (pipeline / vanilla / U-shaped), cut hooks, comm metrics | M4 | `sl/roles.py`, `sl/hooks.py`, `sl/metrics.py` | ⏳ |
| F10 | `ClientSession` (sequential / parallel clients, in-process) | M4, M11 | `sl/clients.py` | ⏳ |
| F11 | Out-of-core training (cpu / disk offload) | M5 | `offload/` | ⏳ |
| F12 | NAT traversal: tunnels, relay | M6 | `transport/tunnel.py` | ⏳ |
| F13 | `ts.Algorithm` API: SplitFed V1/V2, LocalLoss, FedAvg | M8 | `algorithm/` | ⏳ |
| F14 | Fast transport: pinned TCP, CUDA IPC, NCCL | M9 | `transport/` | ⏳ |
| F15 | Client nodes (each client a real node), failures, late joiners | M10 | `executors/client_nodes.py`, `sl/remote.py` | ⏳ |
| F16 | Data: datasets, partitioners (iid/dirichlet/…), evaluation | M11 | `data/` | ⏳ |
| F17 | Privacy: attacks (label, inversion, FSHA, UnSplit, ER), defences, DP | M12 | `privacy/` | ⏳ |
| F18 | Baselines + `SplitServer`/`SplitClient`, Flower interop | M13 | `interop/flower.py`, `baselines/` | ⏳ |
| F19 | Monitoring: OTel tracing / Phoenix, run logs, profiler | M0 | `monitor/` | ⏳ |
| F20 | Config: `RunConfig`, YAML + env overrides | all | `config.py` | ⏳ |

---

## F1. Slicing ⚠️

**What it is.** `ts.slice(model, ...)` turns any `nn.Module` into a `SlicedModel` of consecutive slices.
1. **Trace**: `torch.fx` turns the model into a DAG of named nodes. By default the direct children are the nodes.
   `expand=[...]` (or `"all"`) traces into submodules for finer cuts. Functional ops (`flatten`, `+`, `view`)
   become nodes too. A model fx can't trace raises `TraceError` and points to `pack=` (F2). There is no silent fallback.
2. **Split**: a strategy groups the nodes. `uniform` (node count, default), `params`, `flops`, `profiled`, or
   `cuts=[names]`. Custom strategies subclass `ts.BaseSplitter`.
3. **Validate**: only one tensor may cross each cut, so a cut inside a skip connection is refused.
   `ts.trace(model).summary()` marks the allowed cuts with `cut ok`.

**Example.**
```python
import torchvision, torchslicer as ts
model = torchvision.models.resnet18(num_classes=10)
print(ts.trace(model, expand=["layer3"]).summary())          # node names + "cut ok"
s = ts.slice(model, n=4)                                     # uniform
s = ts.slice(model, strategy="params", n=3)                  # balance parameters
s = ts.slice(model, cuts=["layer2", "layer3.0"], expand=["layer3"])  # cut exactly here
history = s.train(loader, {"name": "SGD", "params": {"lr": 0.05}},
                  {"name": "CrossEntropyLoss", "params": {}}, epochs=1)
```

**Proposed test.** Exactness: on ResNet-18 (residual blocks, BatchNorm, a functional `flatten`), cutting with
each method must give the same training losses as the unsplit model with the same seed and data. Also inspect
the partitions and the bytes crossing each cut. Script: `notes/report/f01_slicing.py`.

**Result (CPU, 3 SGD steps on 8×3×32×32, batch-norm in train mode).**

| Cutting | Partitions | Loss (sliced) | Loss (unsplit) | Bytes fwd per cut (3 batches) |
|---|---|---|---|---|
| `uniform n=4` | conv1–relu / maxpool–layer2 / layer3–avgpool / flatten–fc | 2.108919 | 2.108919 | 1.57 MB, 197 KB, 49 KB |
| `params n=3` | balanced by parameters | 2.108919 | 2.108919 | 98 KB, 49 KB |
| `cuts=[layer2, layer3.0]` | ..layer2 / layer3.0 / layer3.1.. | 2.108919 | 2.108919 | 197 KB, 98 KB |

The difference is 2.5e-7: the mean loss is summed in float32 in a different order. Each cut costs a
different number of bytes: the uniform cut after `relu` sends 16× more than the others. So the choice
of cut matters for SL communication, and the per-cut `bytes_fwd` metric shows it.

**Issue found (usability, minor).** `criterion={"name": "CrossEntropyLoss"}` without `"params"` crashes
with a bare `KeyError: 'params'` (`executors/local.py:236`). It should default to `{}`, and the optimizer config
probably should too. Not fixed yet: your call.

Also already covered by `tests/test_split_anything.py` (~40 tests: every valid cut of ResNet-18 reproduces the
forward, untraceable models raise `TraceError`, validation rules, strategies leave model and RNG untouched).

### F1 follow-up (2026-10-26)

**How to read the graph.** `ts.trace(model)` = `torch.fx` trace grouped at the chosen level: direct
children by default (`model.named_children()` + functional ops such as `flatten`), `expand=["layer1"]`
→ one node per BasicBlock, `expand=["layer1.0"]` → the layers inside the block, `"all"` → every `nn` layer.
`print(model)` shows what the model *contains*; `torch.fx.symbolic_trace(model).code` shows what `forward`
*does* (where the skips are). `<- [...]` = the nodes a node reads; no `cut ok` = cutting there would need two
tensors (e.g. `add <- [8, 3]`: the skip from `maxpool` spans nodes 4–8), or it is the last node.
`g` is only for looking: `ts.slice(model, ...)` traces again with its own `expand=`/`cuts=` (a dotted name in
`cuts=` expands its parent automatically), and refuses invalid cuts with the nearest valid ones.

**Implemented on branch `refinement`:** `g.summary(params=True, example_input=x)` adds per-node parameter
count and size (params + buffers, tied weights counted once and marked `*`, total line), and, with
`example_input`, each node's output shape and size = bytes crossing a cut placed after it. Runs on a copy
(model state and RNG unchanged, checked). Test: `test_summary_sizes_count_tied_weights_once_and_leave_model_untouched`;
documented in `docs/tutorials/custom_strategy.md` §1. ResNet-18 at batch 8, 32×32: cutting after `conv1`
sends 512 KiB, after `layer4` 16 KiB.

**Design notes.**
- Keep the one-tensor-per-cut rule (clean SL contract: bytes, hooks, attacks, transports). Limitation: long
  skips (U-Net) leave almost no valid cut; HF models need `pack=`. Candidate milestone: opt-in multi-tensor cuts.
- Rough edges: functional ops get flat fx names (`add`, `add_1`, not `layer1.0.add`); `expand=` takes names
  only (a class would help: `expand=[BasicBlock]`); criterion/optimizer config without `"params"` → `KeyError`.
- Done on `refinement` (owner: "offer both"): `ts.slice(g, ...)` cuts the graph from `ts.trace` as it is (no
  second trace; `g.model` is kept); `ts.slice(model, expand=..., cuts=...)` still traces on the fly. Given a
  graph, `ts.slice` refuses `expand=`/`pack=` and cuts inside unexpanded modules, naming the `expand=` to use.
  Both paths call the same `trace()` → `ModelGraph.from_module`; a test now guards that they give the same graph.
  Tests: `test_slice_traces_exactly_like_trace`, `test_slice_a_traced_graph_*` (4 new). Tutorial §1 updated.
- Still open: cuts/expand/strategy in the YAML config (option B; today they are Python-only `ts.slice` kwargs).
- Done on `refinement`: `SlicedModel.summary(params=, example_input=)` redesigned. One block per slice, no `...`
  elision; a module wholly inside a slice is shown once (`layer1.0 (7 nodes)`), a module split by a cut node by
  node; `── cut k after <node>: <shape>, <bytes> ──` between slices and `── output ──` at the end (with
  `example_input`); slice size with `params=True`. Test `test_sliced_summary_lists_every_node_*`.
- Full suite before the summary redesign: 452 passed, 1 skipped (7 min). After: split + docs tests 63 passed.

### Config bugs fixed on `refinement` (validation.md 49–51)
Optimizer/criterion dicts: workers forced lr=0.001 when none was given (simulator ≠ cluster:
Adagrad epoch-2 loss 1.3457 local vs 1.3746 centralized); a missing `"params"` crashed some
executors; `RunConfig.create(optimizer="Adam")` crashed. Now `config.normalize_component` checks
the dict once in `train()` and workers use the user's kwargs exactly.

### Open finding: trained weights stay on the workers (simulator ≠ cluster)
After `train()`, the local executor has updated `model` in place; after a centralized or P2P run
(any setting, even vanilla's in-process slice 0, a deserialized copy) `model` still holds its
initial weights; the trained ones are on the workers / in checkpoints (`ts.restore_model_from_run`).
Checked 2026-10-26 on a 3-slice MLP: same loss in all 4 modes, params changed only for local.
Proposed (b): pull the final weights back at teardown, default only the slices the data owner hosts
(`sync_back: hosted | all | none`), so `s.slices()` and evaluating `model` mean the same everywhere.
Waiting for the owner's decision; `s.slices()` waits on it.
