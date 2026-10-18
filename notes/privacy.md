# M12 privacy suite: plan (2026-10-18), steps 1–4 done (2026-10-18)

Goal for adoption: an SL paper's privacy section (attack, defence, leakage metric, accuracy
cost) runs in TorchSlicer from a config, in every setting, with numbers per epoch.

Design:
- **Attacks observe a cut** (the party that receives it: server for activations, client
  for gradients). In the simulator (local executor) the executor also passes the batch's
  true labels/inputs to the attack *for scoring only*. Distributed runs: later (the
  observer has no ground truth there), refused for now.
- `ts.privacy` module: `Attack` (observe(direction, tensor, ctx, truth) → per-epoch
  `metrics()`), `Defence` = existing `CutHook`s (noise, quantize, top-k) plus training-loss
  defences via `ts.Algorithm.loss` (NoPeek). Metrics go into the history as `leak_*`.
- Config: `privacy.attacks: [{name, cut, params}]`.

Steps (each: reproduce the paper's qualitative result on a small setting; negative results
recorded):
1. **Label inference from returned gradients** (Li et al., ICLR 2022: norm and direction
   attacks; metric: AUC). Check: near-perfect AUC on an imbalanced binary task with vanilla
   SL; Gaussian gradient noise lowers it.
2. **Input reconstruction from activations**: learning-based inversion (a decoder trained
   on auxiliary data: PSNR/SSIM, + distance correlation as a model-free metric).
   Check: shallow cut → high PSNR; deeper cut / noise / NoPeek → lower.
3. **NoPeek defence** (distance-correlation loss, arXiv:2008.09161) as an Algorithm,
   and label-gradient noise. Report the accuracy cost next to the leakage.
4. Later (heavier): FSHA (malicious server, CCS 2021), UnSplit model inversion,
   LLM internal-state inversion (Dong et al.), DP accountant.

Status (2026-10-18): steps 1–3 implemented (`torchslicer/privacy/`, tutorial
docs/tutorials/privacy.md, benchmark benchmarks/privacy/, results in experiments.md and
validation.md "M12"). Deviation: NoPeek is a `Defence` (composes with any Algorithm), not an
Algorithm. Open for step 4 and later: FSHA, UnSplit, LLM internal-state inversion, a DP
accountant for GaussianNoise, Marvell; noise relative to the gradient scale (the absolute std
made the MNIST trade-off flat); attacks in distributed runs (observer-side files, scored offline);
a harder label task for a real privacy/utility curve (e.g. CIFAR-10 binary, or Criteo-like tabular).

Status (2026-10-18, later): step 4 implemented: FSHA (`forges`), UnSplit (`needs_client`, fresh),
EmbeddingRecovery (ER, white-box), DPAccountant + ClipNoise (GDP, per-sample rows; not end-to-end).
Results in experiments.md ("M12 step 4"): FSHA partly reproduced at our budget, UnSplit negative,
ER reproduced (depth trend), label DP curve on CIFAR-10 binary. Still open: TBS (deep-layer LLM
inversion), Marvell, FSHA at 1e4 iterations, attacks in distributed runs (observer-side files),
UnSplit's label-inference variant, a fallback from load_dataset("cifar10") to local npz files.
