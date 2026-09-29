# G5 — Training through a party's outage with a stand-in, and merging on rejoin (formalization, 2026-09-29)

Origin: the stand-in/rejoin mechanism prototyped on `research/g1-trust-ft` (`lib/torchslicer/torchslicer/ft/standin.py`,
`executors/trust_recovery.py`). G1's "sealed failover" was judged not novel: recovering inside the same party is leak-free
by construction. Its 3-seed results live in that branch's `notes/experiments.md`. This line keeps only the open part.

## Setting
- Parties P_1..P_n form a chain. Slice f_i(·; θ_i) is owned by P_i, and the model is F = f_n ∘ … ∘ f_1.
  P_1 holds the data and P_n holds the labels.
- h_i = f_i(h_{i-1}) is the activation that P_i sends to P_{i+1}.
- θ^(0) are the initial weights. In TorchSlicer they are shipped by the coordinator, so they are public to it and could be
  shared with every party.
- **Outage.** A middle party P_k (1<k<n) fails at step t_0 and returns at t_1 = t_0 + T. Its last snapshot is θ_k^snap,
  taken at step s ≤ t_0.
- **Constraints**, which hold throughout:
  - (C1) trained θ_k never leaves P_k in plaintext;
  - (C2) no party receives activations from a shallower cut than before the outage.

## Stand-in phase (t_0 ≤ t < t_1)
- P_{k-1} hosts a stand-in g(·; φ) with the same architecture as f_k. It computes h_k' = g(h_{k-1}; φ) and forwards it.
  Trained θ_k is never used.
- Initialisation φ_{t_0} ∈ {random, θ_k^(0) (shared public init), public pretrained weights}.
- Training minimises the task loss L over (θ_1..θ_{k-1}, φ, θ_{k+1}..θ_n). Optional additions:
  - an interface loss R(φ) = D(stats(h_k'), s_k), where s_k are low-dimensional statistics of h_k (e.g. per-channel
    mean/var) that P_{k+1} collected before the outage;
  - downstream damping: a reduced LR, or freezing, for θ_{>k}.
- Effects to quantify:
  - **(E1) capacity gap:** φ is undertrained relative to θ_k;
  - **(E2) co-adaptation:** θ_{>k} drift toward the distribution of h_k'.

## Rejoin (t = t_1)
P_k returns with θ_k^snap. The stand-in state is φ_{t_1}, and the downstream parties have adapted. A **merge operator** M
produces the slice P_k resumes with:

  θ_k^new = M(θ_k^snap, φ_{t_1}, θ^(0), info)

**Merge must run at P_k.** Otherwise θ_k leaks, violating C1.

Candidates:
| id | operator | notes |
|---|---|---|
| M0 replace | θ_k^new = θ_k^snap, φ discarded | current prototype; baseline |
| M1 keep | θ_k^new = φ_{t_1} | discards the pre-outage training |
| M2 step-avg | α θ_k^snap + (1−α) φ_{t_1}, α = s / (s + T) | the naive proposal; expected to fail when inits differ (neurons not aligned) and when training stages differ |
| M3 aligned-avg | permute φ to align with θ_k^snap, then average | model-merging / re-basin style [to retrieve] |
| M4 delta | θ_k^snap + λ (φ_{t_1} − θ_k^(0)) | needs φ_{t_0} = θ_k^(0); stale-update view, akin to async FL [to retrieve] |
| M5 functional hand-over | for H steps, P_{k-1} still sends g(h_{k-1}); P_k outputs β_t f_k(h_{k-1}) + (1−β_t) g(h_{k-1}), with β_t: 0→1, and optionally distils f_k toward g | no weights exchanged, only activations; smooth for downstream |

## Exposure question
What does P_k learn from what it receives?
- In M1–M4 it receives φ_{t_1}.
- In M5 it receives g-outputs.

φ was trained only on inputs and gradients that P_k would have seen at its own position had it not failed. Hypothesis: the
exposure is ≤ normal participation. But updates can leak training data (gradient-inversion literature [to retrieve]), so it
must be measured, e.g. by reconstruction or membership inference on φ_{t_1} − φ_{t_0} vs on P_k's normal view.

## Research questions / hypotheses
- **RQ1 (cost of outage):** the accuracy gap during the outage, and the final accuracy gap, grow with T and with slice
  importance. H: noticeable on CIFAR-100 / GPT-2, unlike CIFAR-10 (where G1 saw no effect).
- **RQ2 (mitigation during outage):** public or shared init, interface loss R, and downstream damping reduce the gap
  (area between the stand-in and no-failure accuracy curves).
- **RQ3 (merge):** M2 ≤ M0 when inits differ, and M3/M4/M5 > M0 on rejoin dip and final accuracy.
  Metrics: loss spike at t_1, steps to recover, final acc.
- **RQ4 (exposure):** exposure of the transferred object at P_k ≤ its normal-participation view, measured.

## Baselines
- No failure.
- Pause: stall training until the rejoin (no stand-in).
- Stand-in + M0.
- Leaky upper bound: the neighbour takes over the trained θ_k (FTPipeHD-style), which gives accuracy but violates C1/C2.

## Metrics (per run, ≥3 seeds)
- Test-accuracy curve.
- Outage gap = area vs no-failure.
- Rejoin dip = max loss increase in the window after t_1, and steps to return to the pre-rejoin level.
- Final accuracy.
- Extra compute and bytes at the host and at P_k.
- Exposure (attack metric on transferred objects + inversion at each party's input).

## Settings
1. ResNet-18 on CIFAR-100; slices as in G1 (D | layer2 | layer3 | L); T ∈ {1, 3, 5} epochs.
2. Vary slice importance: a larger middle slice.
3. GPT-2 small on a text task (packed blocks; stand-ins are natural there).

## First cheap check (no new training)
In G1's existing stand-in + rejoin runs, check whether the per-batch training loss spikes at the rejoin batch. That
signals co-adaptation (E2). Data: `results/g1/runs/g1_pilot_P2-standin-rejoin_s{0,1,2}/batch_metrics.jsonl` on branch
`research/g1-trust-ft` (rejoin batch is in `train_summary.json` → ft_events). The run directories are local, not in git.
