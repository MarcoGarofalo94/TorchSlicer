# TorchSlicer — Abstractions Roadmap

Tracking branch: `feat/abstractions`
Goal: position TorchSlicer as the canonical split learning framework (inspired from Flower for FL).

---

## Phase 1 — Zero-config Entry Point

| Task | Status |
|---|---|
| `ts.run(model, loader, n=2)` one-liner via `LocalExecutor` | ✅ done |
| String shorthands for optimizer/criterion in `SlicedModel.train()` | ✅ done |
| `ts.simulate()` — multi-process local simulation without Docker | ✅ done |

**`ts.simulate()` spec:**
- Spawn N worker processes via `torch.multiprocessing.spawn`
- Reuse `DistributedExecutor` + `WorkerServicer`, bind to localhost ports automatically
- No config file or Docker required
- Target API: `ts.simulate(model, loader, n=4, devices=["cpu","cpu","cpu","cpu"])`

---

## Phase 2 — Strategy Abstraction

| Task | Status |
|---|---|
| `ParameterBalancedSplitter` — split by trainable parameter count | ✅ done |
| `ExplicitSplitter` — user-specified layer boundary indices | ✅ done |
| `@ts.register_strategy` decorator (public, documented) | ✅ done |
| Pipeline schedule abstraction (`BasePipelineSchedule`) | ✅ done |
| Refactor GPipe into `pipeline/gpipe.py` (extracted from executor) | ✅ done |
| `PipeDreamSchedule` (async 1F1B) | ⏭ deferred — needs weight stashing, blocked on 4+ GPU benchmark showing bubble is bottleneck |

**Notes on strategies:**
- `ParameterBalancedSplitter` and `ExplicitSplitter` are straightforward and low-risk.
- Paper-backed strategies (e.g. cost-model partitioning from PipeDream, Alpa, etc.) are deferred
  until we have a clear citation target — no random implementations.
- Pipeline schedule abstraction is medium effort but unlocks research extensibility.

---

## Phase 3 — HuggingFace Integration

| Task | Status |
|---|---|
| `ts.hf_pack(model_type)` registry lookup | ✅ done |
| Pack functions moved into library (`adapters/hf_packs.py`) | ✅ done |
| `ts.from_pretrained(name_or_model, n, task, ...)` | ✅ done |
| Auto-detect pack function from `model.config.model_type` | ✅ done |
| PEFT/LoRA first-class path (auto-unwrap in from_pretrained) | ✅ done |

**PEFT spec:**
- `ts.from_pretrained(peft_model, n=4)` should auto-call `peft_unwrap` internally
- Add integration test with a toy LoRA-wrapped model (no real HF download)

---

## Phase 4 — Research Extensibility

| Task | Status |
|---|---|
| Topology abstraction (`BaseSplitTopology`) | ✅ done |
| U-shaped split topology | ✅ done |
| Vertical FL split topology | ⏭ deferred — needs cross-feature split support (different input per party), out of scope for now |
| Activation hook interface (`on_forward_smash`, `on_backward_smash`) | ✅ done |
| NoPeek privacy hook (distance correlation minimization) | ✅ done |
| Differential privacy noise injection hook | ✅ done |

---

## Phase 5 — Community Content

| Task | Status |
|---|---|
| Paper-reproducible example: vanilla SplitNN (Vepakomma et al. 2018) | ⬜ todo |
| Paper-reproducible example: U-shaped split | ⬜ todo |
| Paper-reproducible example: NoPeek defense | ⬜ todo |
| Model zoo: ResNet-18 CIFAR-10 (already exists, needs cleanup) | ⬜ todo |
| Model zoo: BERT SST-2 (already exists, needs cleanup) | ⬜ todo |
| Benchmarking suite (`python -m torchslicer.benchmark`) | ⬜ todo |
| Auto-generated benchmark table (replaces hand-written one in CLAUDE.md) | ⬜ todo |

---

## Version milestones

| Version | Milestone |
|---|---|
| `v0.3.0` | `ts.simulate()`, PEFT auto-unwrap, `ParameterBalancedSplitter`, `ExplicitSplitter` |
| `v0.4.0` | Pipeline schedule abstraction, topology variants (U-shaped, vertical) |
| `v0.5.0` | Activation hooks, privacy hooks (NoPeek, DP noise) |
| `v0.6.0` | Paper-reproducible examples, benchmarking suite, model zoo cleanup |
| `v1.0.0` | API stable, community validated |
