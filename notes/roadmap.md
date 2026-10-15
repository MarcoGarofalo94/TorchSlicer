# TorchSlicer roadmap (2026-10-14)

Goal: become the de facto framework for split learning (SL) experiments, the way Flower is for FL.
TorchSlicer is SL-only. Averaging/FL logic stays out of the core; it plugs in through hooks and is
shown in a Flower-based SplitFed baseline.

## Pillars
1. **Split anything**: automatic slicing via `torch.fx`, named cuts, strategies, and `pack=` when tracing can't handle the model.
2. **Run anywhere**: the same experiment runs in local simulation, Docker, bare metal or k8s; centralized *or* P2P orchestration.
3. **Extend everything**: documented, registered extension points (splitter, cut hook, orchestration, transport, discovery, callbacks).
4. **Community baselines**: curated, reproducible SL papers contributed via a template (cf. Flower Baselines).

## Status
- M0 (branch `feat/m0-foundations`, not merged/pushed): done = metrics fix, persistent workers, CI, CITATION/Zenodo metadata, pyproject sync.
  Open = license decision; untrack the stale `lib/torchslicer/build/`; enable Zenodo's GitHub integration; rebuild/push the Docker image.
- Model x mode check (2026-10-15): DNN, CNN, ResNet-18, TinyGPT, GPT-2, Llama (+ LoRA on the 3 LMs) x
  local / GPipe / centralized gRPC+TCP / P2P gRPC+TCP, CPU (SGD, AdamW) and A100 (AdamW), compared with unsplit PyTorch
  under the same seed: all match. The one known difference is GPipe + BatchNorm, which micro-batching causes by
  design and should be documented. Fixed on M0: tied weights, float32 lr, 5 s worker-init deadline.
  Carried to M2: a crashed P2P follower hangs the driver (no failure detection); `examples/train/lm/{worker,lora_worker}`
  duplicate the P2P driver (no tied-weight support); turn the matrix harness into an integration test (+ small CI subset).

## Milestones (one `feat/m<k>-<slug>` branch each, from `main`)
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

## Known facts this is based on (verified in code, 2026-10-14)
- Centralized: the coordinator holds no layers and sends raw inputs to W0 → raw data leaves the data owner (fixed by M4 roles).
- P2P: the driver is worker 0 plus an embedded coordinator; it's an example script with no heartbeats/recovery.
- Workers exit after every run: `DistributedExecutor.teardown()` sends `shutdown` to all of them, idle ones included.
- A centralized worker exits(1) if it can't register within ~30 attempts × 3 s.
- No U-shaped / multi-client / SplitFed in the library; `uniform` is the only built-in splitter.
- SplitBud (Flower 1.9 fork, 2-tier, manual split, outbound-only clients) and SLPerf (MPI, manual split): see `sota.md`.
