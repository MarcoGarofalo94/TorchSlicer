# TorchSlicer roadmap (created 2026-10-14, updated 2026-10-15)

Goal: become the de facto framework for split learning (SL) experiments, the way Flower is for FL.
TorchSlicer is SL-only. Averaging/FL logic stays out of the core; it plugs in through hooks and is
shown in a Flower-based SplitFed baseline.

## Pillars
1. **Split anything**: automatic slicing via `torch.fx`, named cuts, strategies, and `pack=` when tracing can't handle the model.
2. **Run anywhere**: the same experiment runs in local simulation, Docker, bare metal or k8s; centralized *or* P2P orchestration.
3. **Extend everything**: documented, registered extension points (splitter, cut hook, orchestration, transport, discovery, callbacks).
4. **Community baselines**: curated, reproducible SL papers contributed via a template (cf. Flower Baselines).

## Status
- M0 (branch `feat/m0-foundations`, not merged/pushed): **done** = metrics fix, persistent workers, CI, CITATION/Zenodo metadata, pyproject sync,
  tied-weight / lr / init-timeout fixes, equivalence harness + `notes/validation.md`.
  Still open (owner decisions) = license decision; untrack the stale `lib/torchslicer/build/`; enable Zenodo's GitHub integration; rebuild/push the Docker image.
- Model x mode check (2026-10-15): DNN, CNN, ResNet-18, TinyGPT, GPT-2, Llama (+ LoRA on the 3 LMs) x
  local / GPipe / centralized gRPC+TCP / P2P gRPC+TCP, CPU (SGD, AdamW) and A100 (AdamW), compared with unsplit PyTorch
  under the same seed: all match (full record: `notes/validation.md`, harness `benchmarks/equivalence/`). The one known difference is GPipe + BatchNorm, which micro-batching causes by
  design and should be documented. Fixed on M0: tied weights, float32 lr, 5 s worker-init deadline.
  Carried to M2: a crashed P2P follower hangs the driver (no failure detection); `examples/train/lm/{worker,lora_worker}`
  duplicate the P2P driver (no tied-weight support); turn the matrix harness into an integration test (+ small CI subset).

## Branching: incremental, one branch per milestone
- Each milestone gets one branch, `feat/m<k>-<slug>`, forked from the **tip of the previous milestone's branch**, not
  from `main`. The chain is `main` → `feat/m0-foundations` → `feat/m1-split-anything` → `feat/m2-...` and so on.
  `main` stays untouched until the owner decides to merge, which happens in order (M0 first).
- One agent session owns one milestone. When it finishes: the full test suite passes, the "Status" and "Handoff"
  sections here are updated, issues and fixes are recorded in `notes/validation.md`, and everything is committed on the
  milestone branch. Don't push, and don't merge into `main`, unless the owner asks.
- If the previous milestone branch gets new commits after the next one forked from it, rebase or merge them forward
  and say so in the commit message.

## Milestones
- **M0, foundations**: workers reusable across runs (end-of-run shutdown optional); CI running the tests on every PR;
  `CITATION.cff` + Zenodo metadata; merge `fix/local-epoch-metrics`; license decision (GPL-3.0 vs Apache-2.0, owner's call).
- **M1, split anything**: named cuts; param/FLOPs/profiled strategies; tutorial "write your own cutting strategy";
  a clear fx-failure error pointing to `pack=` + pack tutorial.
- **M2, P2P as a reference topology**: move it from `examples/train/p2p/worker/main.py` into the library, sharing the servicer
  and transport with centralized. Keep it basic, and document the extension points (discovery, plan agreement, scheduling,
  failure handling) with one worked example that replaces one of them.
- **M3, run anywhere**: one config gives local / compose / bare metal / k8s (StatefulSet = stable ordered DNS names,
  Job for the coordinator/driver); one deployment guide page per environment.
- **M4, SL settings**: roles (the data owner hosts stage 0 so raw data stays local; label owner; U-shaped); `CutHook`
  (compression, quantization, noise, attack observers); built-in bytes-per-cut / client-compute / simulated-link metrics;
  multi-client sessions (sequential/parallel) with a client-weight handoff hook, no averaging in the core.
- **M5, NAT-friendly transport**: the gradient comes back in the reply (clients need only outbound connections); optional relay.
- **M6, baselines + paper**: `baselines/` template (config, seeds, expected numbers, CI smoke test, mean ± std over ≥3 seeds);
  seed it with vanilla SplitNN, U-shaped, SplitFed-via-Flower, and one compression/attack paper; framework paper vs SplitBud/SLPerf.

## Handoff (read this first if you are a new session)
Last updated 2026-10-15, at the end of M0.
- **Where things stand:** M0 is done on `feat/m0-foundations`. **Next: M1.** Create
  `git switch -c feat/m1-split-anything feat/m0-foundations`.
- **Read first:**
  - this file;
  - `notes/validation.md` (how correctness is checked, and every bug found so far);
  - `CLAUDE.md` (layout, commands, architecture).
  - `notes/sota.md` and `notes/gaps.md` are for the *research* lines (`research/*` branches), not for framework milestones.
- **Every change must keep:**
  - `.venv/bin/python -m pytest tests/` green (48 tests at the end of M0);
  - the equivalence matrix green: `.venv/bin/python benchmarks/equivalence/run_matrix.py` (about 15 min on CPU;
    pass a model name for a quick check). Set `MATRIX_DEVICE=cuda` for the GPU, and
    `MATRIX_OPT='{"name":"AdamW","params":{"lr":0.001,"weight_decay":0.01}}'` for AdamW. A change to splitting
    must not change any loss.
- **Record as you go:** each bug found or behaviour changed goes into `notes/validation.md`, with the symptom, root
  cause, fix, cost and test, in the same commit as the fix. Note the pre-fix numbers *before* rerunning, because logs
  get overwritten. A paper on TorchSlicer will be written from these notes.
- **Environment:**
  - Git commits: the container has no git identity, so commit with `git -c user.name=... -c user.email=...`. Ask the
    owner for the values if your memory doesn't have them.
  - Downloads: HF hub and dataset downloads are blocked. Use random-init HF configs and synthetic data, as in
    `benchmarks/equivalence/tsmodels.py`.
  - gRPC stubs: regenerating them needs grpcio-tools 1.64.0 (so the gencode matches the declared minimum), not the
    venv's newer one.
  - Don't commit the owner's untracked files at the repo root (`*.prompt.md`, `prompt*.txt`, `results/`,
    `sota_check.md`) or their `.gitignore` change.
- **Facts for M1 (verified in code, 2026-10-15):**
  - `ModelGraph.from_module()` (`core/model_graph.py`) traces with a *shallow* fx tracer: direct children of the root
    are leaves. So ResNet-18 becomes about 10 nodes (`layer1`..`layer4` are atomic), and cut granularity is limited to
    top-level children.
  - If tracing fails, it **silently** falls back to `from_sequential()`, which treats children as a chain. That gives
    a wrong graph for non-sequential models, with no error. This is the "clear fx-failure error" item.
  - `strategies/`: `BaseSplitter` (`split`, `validate`), a name registry, and `uniform` as the only splitter (it
    balances layer *count*). `validate()` requires exactly one tensor to cross each boundary, so skip connections
    can't straddle a cut.
  - `ts.slice(model, strategy=..., n=..., pack=...)`; `pack=` skips tracing (`ModelGraph.from_stages`).
  - Tied parameters across cuts are handled (`core/tied.py`), so new splitters don't need to care, as long as
    partitions are lists of graph-node indices.
- **Open owner decisions** (don't decide these alone): license GPL-3.0 vs Apache-2.0; untracking the stale
  `lib/torchslicer/build/`; when to merge into `main` and push; Zenodo/GitHub integration; rebuilding the Docker image.

## Known facts this is based on (verified in code, 2026-10-14)
- Centralized: the coordinator holds no layers and sends raw inputs to W0 → raw data leaves the data owner (fixed by M4 roles).
- P2P: the driver is worker 0 plus an embedded coordinator; it's an example script with no heartbeats/recovery.
- Workers exit after every run: `DistributedExecutor.teardown()` sends `shutdown` to all of them, idle ones included.
- A centralized worker exits(1) if it can't register within ~30 attempts × 3 s.
- No U-shaped / multi-client / SplitFed in the library; `uniform` is the only built-in splitter.
- SplitBud (Flower 1.9 fork, 2-tier, manual split, outbound-only clients) and SLPerf (MPI, manual split): see `sota.md`.
