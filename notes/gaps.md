# Candidate gaps (2026-09-28) — awaiting pick

Recurring pattern in the literature: **multi-hop / chained SL is studied only via simulation or single-machine emulation**, and **no SL work handles loss of an intermediate node**. TorchSlicer already runs real N-hop chains (gRPC/TCP, GPipe, P2P, re-slicing recovery, HF LLM/MoE + LoRA). That's where it's strongest.

## G1 — Trust-constrained fault tolerance for multi-hop SL  ★ recommended
- Evidence: Hara & Sasabe 2604.25197 / 2509.10001 and Wei et al. 2505.04368 give no failure handling; the LLM-SL survey 2604.24468 lists none; SFL robustness work (2510.21155, 2509.17398) only covers *client* dropout. Oobleck/Bamboo handle failures but assume any stage can move to any node, which doesn't hold in SL: the data owner has to keep the input slice and the label owner the loss slice, and moving a slice exposes its weights and its activation depth to a new party.
- Gap: recovery that respects **ownership/trust constraints** (pinned slices, who may host which layers, and not handing a node a shallower cut).
- Current TorchSlicer: epoch-level checkpoint + uniform re-slice over survivors (`executors/distributed.py:917`), which ignores ownership.
- H1: constraint-aware recovery (neighbor absorption / redundant shadow slice / batch-level checkpoints) gets goodput within X% of Oobleck-style unconstrained re-slicing, with zero ownership violations, and cuts lost work vs epoch-restart by ≥5×.
- Metrics: time-to-recover, lost samples, goodput (samples/s) under a failure trace, bytes moved at recovery, # ownership violations, and final accuracy parity.

## G2 — Sim-to-real gap of multi-hop SL cost models
- Evidence: the MSP/pipelining papers optimize analytic latency models (compute + size/bandwidth) and validate in simulation. TorchSlicer's own README shows serialization overhead of 1.5–3.5× over local, which those models leave out.
- H2: on a real N-hop deployment, analytic models mispredict per-batch latency by ≥30% and choose a suboptimal cut/micro-batch; a profile-calibrated model (TorchSlicer profiler + OTel traces) closes the gap. It's a measurement paper plus a calibrated splitter.
- Risk: less novel on its own, but a good systems/benchmark track paper and a natural companion to G1.

## G3 — Per-hop privacy leakage and collusion in chained SL
- Evidence: Split-n-Chain claims that more intermediate nodes make attacks harder, but the abstract shows no attack evaluation. Dong et al. (USENIX Sec '25) show depth doesn't protect LLM states. Politis & Pappas show that the gradient channel leaks what forward checks miss.
- H3: in an N-hop chain, (a) a middle hop that sees both activations and returning gradients inverts inputs much better than forward-only analysis predicts, and (b) k colluding non-adjacent hops recover nearly as much as a single 2-tier server. So "multi-hop = more private" is false without defenses.
- Metrics: reconstruction (PSNR/SSIM, token F1) per hop position, k-collusion, cost of defenses (bytes, accuracy).

## G4 — Multi-hop split LoRA fine-tuning of LLM/MoE on real heterogeneous hosts
- Evidence: SplitLoRA and JTO are 2-tier SFL. The survey says LLM SL lacks native-task, real-system validation.
- H4: expert/block placement across hops (MoE-aware splitting) reduces the bottleneck stage time vs uniform splitting at equal quality.
- Risk: crowded area, and it needs GPUs.

## Suggested paper shape
"Resilient multi-hop split learning": G1 as the method, G2 as the realistic evaluation substrate, and optionally G3 as the trust model that motivates G1's constraints. TorchSlicer becomes the artifact rather than the claim.

## G5 — Training through a party's outage with a stand-in, and merging on rejoin (added 2026-09-29)
- Formalized in `notes/proposal_g5_standin_merge.md`. Comes out of G1: its "sealed failover" was judged not novel
  (recovery inside the same party is leak-free by construction). The open part is the accuracy cost of stand-ins and how
  to merge the stand-in's learning into the owner's slice at rejoin without exposing weights.
- H: the step-weighted weight average fails when inits differ; delta-merging from a shared public init or a functional
  hand-over reduces the rejoin dip and the final accuracy gap vs discarding the stand-in, at no more exposure than normal
  participation.
