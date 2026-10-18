# SOTA — Split Learning (retrieved 2026-09-28)

Format: title — year, venue — link — idea — setting — numbers. `[abstract-only]` = only abstract/summary read.

## Frameworks / benchmarks (direct competitors to TorchSlicer)
- **SplitBud: Towards a Unified Framework for Split Learning** (Radovič, Canini, Horváth, Pejović, Vepakomma) — 2025, EuroMLSys — https://doi.org/10.1145/3721146.3721936 , code https://github.com/sands-lab/splitbud — Flower extension; client↔server "request" abstraction, gRPC unary/streaming, implements SplitFed v1/v2 etc. States "most SL algorithms evaluated only in simulation", "no framework to easily design, evaluate and deploy SL". Setting: 2-tier client/server, multi-client. No multi-hop chain, no fault tolerance described.
- **SLPerf** (Zhou, Hu, Wu, Chen) — 2023, arXiv — https://arxiv.org/abs/2304.01502 — benchmark library of 13 SL variants (vanilla, U-shape, PSL, AsyncSL, SplitFed, FSL, HSFL, CPSL, ...). Simulated distribution; MNIST/CIFAR-10/Adult/ogbg-molhiv, IID + Dirichlet non-IID. SplitFed 97.68% MNIST at α=0.1.

## Surveys
- **A Survey on Split Learning for LLM Fine-Tuning: Models, Systems, and Privacy Optimizations** (Liu, Wang, Wang, Tang, Wu) — 2026, arXiv — https://arxiv.org/abs/2604.24468 — open problems listed: server memory bottleneck, stragglers, compression, non-IID, auto-regressive label leakage, pipeline bubbles under heterogeneity, **no discussion of failure recovery / node departure**, lack of empirical validation on native LLM tasks.
- **A Taxonomy of Attacks and Defenses in Split Learning** — 2025, arXiv — https://arxiv.org/abs/2505.05872 `[abstract-only]` — notes semi-honest assumptions underestimate collusion/gradient-manipulation risk.

## Multi-hop / pipelined SL (closest to TorchSlicer's N-slice chain)
- **Optimization of Model Splitting, Placement, and Chaining for Multi-hop SL and Inference** (Hara, Sasabe) — 2026, arXiv — https://arxiv.org/abs/2604.25197 — ILP + BCD heuristic for split/placement/routing. **Simulation only** (NSFNET 14 nodes, ResNet101, 1 Gbps). BCD ≈ ILP objective, ~100× faster. No failures, no runtime re-split.
- **Service Function Chaining Architecture for Multi-hop Split Inference and Learning** (Hara, Sasabe) — 2025, arXiv — https://arxiv.org/abs/2509.10001 — SRv6/eBPF chaining of model slices. Mininet emulation on one machine, reduced ResNet101 in 4 slices, CIFAR-100. Training 173 ms/round, no gain over baseline; reroutes around congestion; **no node-failure handling**.
- **Pipelining Split Learning in Multi-hop Edge Networks** (Wei, Lin, Li, Li, Chen) — 2025, IEEE TMC — https://arxiv.org/abs/2505.04368 — pipeline parallelism + model split/placement as bottleneck-aware shortest path; closed-form micro-batch size. **"Extensive simulations"** only.
- **PiPar: Pipeline Parallelism for Collaborative ML** — 2024, JPDC — https://arxiv.org/abs/2302.12803 `[abstract-only]` — pipelines server/device compute + comms; up to 34.6× faster than FL, server idle time ↓64.1×. 2-tier.
- **Split-n-Chain** (Sahani, Sengupta) — 2026, Cluster Computing — https://arxiv.org/abs/2503.07570 `[abstract-only]` — splits layers among several intermediate nodes claiming it is "harder to compromise multiple nodes" + blockchain audit. No empirical per-node attack evaluation visible in abstract.

## Split LLM fine-tuning
- **SplitLoRA** — 2024, arXiv — https://arxiv.org/abs/2407.00952 `[abstract-only]` — first SFL+LoRA LLM fine-tuning framework; client-side LoRA aggregated.
- **Just Talk Once: Communication-Efficient Split Federated LLM Fine-Tuning on Edge Devices** — 2026, arXiv — https://arxiv.org/abs/2609.01457 `[abstract-only, via search snippet]` — real Wi-Fi 6 testbed, A800 server + Jetson Orin Nano clients.
- **Unveiling the Depth-Performance Dilemma in Split-Federated Fine-tuning of LLMs** — 2026, arXiv — https://arxiv.org/abs/2608.22188 `[unverified: title only]`.

## Robustness / stragglers in SL
- **MU-SplitFed: Towards Straggler-Resilient SFL: An Unbalanced Update Approach** — 2025, NeurIPS — https://arxiv.org/abs/2510.21155 `[abstract-only]` — decouples server progress from straggler clients. 2-tier SFL.
- **Optimizing SFL with Unstable Client Participation** — 2025, arXiv — https://arxiv.org/abs/2509.17398 `[abstract-only]` — upload failures hurt most, download failures least. Client dropout only (not intermediate-server loss).

## Fault-tolerant pipeline parallelism (datacenter baselines, not SL)
- **Oobleck** — 2023, SOSP — https://arxiv.org/abs/2309.08125 — pipeline templates, f+1 replicas; up to 13.9× over Bamboo/Varuna. Assumes stages are freely movable across nodes.
- **Bamboo** `[unverified: via Oobleck summary]` — redundant computation of successor stage.
- **ReCycle** — 2024, SOSP — https://doi.org/10.1145/3694715.3695960 `[title only]`.

## Privacy of intermediate representations
- **Depth Gives a False Sense of Privacy: LLM Internal States Inversion** (Dong et al.) — 2025, USENIX Security — https://arxiv.org/abs/2507.16372 — inverts middle-layer states; 4,112-token prompt recovered at 86.88 token F1 from middle layer of Llama-3; 4 defenses insufficient. Full text read (2026-10-18, §4.2, §5.2): white-box
  ER (Embedding Recovery) optimizes dummy input embeddings w^ (init 0) so the first l layers reproduce h_l, keeps the
  min-loss iterate, and maps each position to the vocab embedding of highest cosine; TBS (Token Basis Selection) is the
  deep-layer variant (optimizes coefficients on an SVD basis of the embedding matrix). Defaults: AdamW, MSE, lr 5e-4,
  penalty λ = 0, E = 5e4 steps; also black-box (transfer to derived models) and generation-based (inversion model).
  Implemented as ts.privacy.EmbeddingRecovery (ER only, 2026-10-18).
- **Privacy Failure in Split-LLM Training: The Returned Gradient Nullifies the Decoys** (Politis, Pappas) — 2026, arXiv — https://arxiv.org/abs/2609.04382 — decoy rows get zero gradient → 4096/4096 real rows identified; forward-only privacy checks miss gradient channel. Vanilla 2-node.
- **UnSplit** — 2022, WPES — https://eprint.iacr.org/2021/1074 `[abstract-only]` — honest-but-curious server recovers inputs + client model knowing only architecture.
- **Model Inversion Attacks Against Collaborative Inference** (He, Zhang, Lee) — 2019, ACSAC — https://doi.org/10.1145/3359789.3359824 (program: https://acsac.org/2019/program/final/s167.html; code: https://github.com/zechenghe/Inverse_Collaborative_Inference) `[abstract-level + repo README; full text not read]` — a malicious participant of a split network recovers inputs: white-box rMLE, black-box *inverse network* (a decoder from cut features to inputs), and a query-free attack (shadow model); CIFAR-10, worst when the client keeps few layers. The inverse network's training data (attacker queries with same-distribution samples) is `[unverified]`. Basis of M12's `InversionAttack`.
- **NoPeek: Information leakage reduction to share activations in distributed deep learning** (Vepakomma, Singh, Gupta, Raskar) — 2020, arXiv (venue not stated) — https://arxiv.org/abs/2008.09161 `[abstract-only]` — adds a loss term minimizing the distance correlation between raw inputs and the shared cut activations, against reconstruction attacks, while keeping accuracy. No numbers in the abstract.

## Differential-privacy accounting (retrieved 2026-10-18, for M12 step 4)
- **Gaussian Differential Privacy** (Dong, Roth, Su) — 2019, arXiv (journal version not checked `[unverified: venue]`) —
  https://arxiv.org/abs/1905.02383 — μ-GDP via hypothesis testing. Thm 2.7: the Gaussian mechanism with std
  sens/μ is μ-GDP; Cor. 3.3: n-fold composition of μi-GDP is sqrt(Σμi²)-GDP; Cor. 2.13: μ-GDP ⇔ (ε, δ(ε))-DP with
  δ(ε) = Φ(−ε/μ + μ/2) − e^ε Φ(−ε/μ − μ/2). Used by ts.privacy.DPAccountant.
- **Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising** (Balle,
  Wang) — 2018, ICML (stated on the arXiv PDF) — https://arxiv.org/abs/1805.06530 — Thm 8: the Gaussian
  mechanism with L2 sensitivity Δ and std σ is (ε, δ)-DP iff Φ(Δ/2σ − εσ/Δ) − e^ε Φ(−Δ/2σ − εσ/Δ) ≤ δ (the same
  curve as GDP Cor. 2.13 with μ = Δ/σ); the classical σ = Δ sqrt(2 ln(1.25/δ))/ε is loose.

## Leakage from weight updates (copied from the research branches, 2026-10-18)
- **Deep Leakage from Gradients** (Zhu, Liu, Han) — 2019, NeurIPS `[unverified: venue]` — https://arxiv.org/abs/1906.08935 — pixel-/token-accurate reconstruction from shared gradients.
- **Inverting Gradients** (Geiping, Bauermeister, Dröge, Moeller) — 2020, NeurIPS `[unverified: venue]` — https://arxiv.org/abs/2003.14053 — cosine-loss inversion; averaging over several iterations or images does not protect privacy.
- **Data Leakage in Federated Averaging** (Dimitrov, Balunović, Konstantinov, Vechev) — 2022, TMLR `[unverified: venue]` — https://arxiv.org/abs/2206.12395 — inverts multi-epoch FedAvg weight deltas: >45% of images recovered from 10 local epochs × 10 batches × 5 images (FEMNIST) vs <10% for baselines. **Implication for RQ4:** φ_{t1} − φ_{t0} is a multi-step weight delta over P_{k-1}'s outputs and B's gradients, so it is a known leakage channel. Shipping it to P_k is not obviously ≤ normal participation (see gaps.md).

## Failures/dropouts as an attack surface; active and label attacks in SL (retrieved 2026-09-29, for the G5 reframing)
- **Practical Secure Aggregation for Privacy-Preserving ML** (Bonawitz et al.) — 2017, CCS `[unverified: venue, not on the pages read]` — https://eprint.iacr.org/2017/281 `[full-text §4.0.2–4.0.3]` — **closest precedent for fake failures.** Two quotes. A slow user is assumed dropped, the others reveal shares of its key, then its late message arrives and the server "learn[s] x_u in the clear". And "an adversarial server in the active model can similarly learn x_u simply by lying about whether user u has dropped out." Fix: double masking plus a signed ConsistencyCheck round. Lesson for us: make the protocol safe *even if* the failure claim is false; don't only harden detection. FL/SecAgg, a single server, no chain.
- **Eluding Secure Aggregation in FL via Model Inconsistency** (Pasquini, Francati, Ateniese) — 2022, CCS — https://arxiv.org/abs/2111.07380 — a malicious server sends different models to different users and bypasses SecAgg. Protocol-level manipulation beats the cryptographic protection.
- **DROPFL: Client Dropout Attacks Against FL Under Communication Constraints** (Qian et al.) — 2024, IEEE (ICASSP?) `[abstract-only via search snippet; unverified: venue]` — https://ieeexplore.ieee.org/document/10446609/ — controlled clients drop out deliberately: up to −10.2% accuracy, up to 4.1× training time. The goal is **availability/accuracy, not privacy.**
- **Unleashing the Tiger: Inference Attacks on Split Learning (FSHA)** (Pasquini, Ateniese, Bernaschi) — 2021, CCS `[venue via search snippet]` — https://arxiv.org/abs/2012.02670 `[full text read 2026-10-18; venue still unverified, not in the arXiv PDF]` — a malicious server uses its control of the gradients to steer the client's feature space into a known invertible one, then reconstructs inputs; also shows malicious clients can attack honest ones. **Relevance:** during an outage the downstream party supplies all gradients that train a *freshly initialised* stand-in, the easiest possible target for hijacking. Full text read (2026-10-18, §3.1–3.3, App. B): the server
  trains a pilot f~ and inverse f~^-1 as an auto-encoder on X_pub (Eq. 1, MSE), a critic D between f~(X_pub) and
  f(X_priv) (Eq. 2; implementation: Wasserstein loss + gradient penalty, weight 500), and sends the client the gradient
  of log(1 − D(f(X_priv))) (Eq. 3) instead of the task's. Client f: ResNet funnel, splits 1–4; pilot = shallow linear
  convs; D deeper residual. Adam lr 1e-5 (f, f~, f~^-1), 1e-4 (D). MNIST/F-MNIST reconstruct within ~1e3 setup
  iterations (MSE in [−1, 1] ≈ 0.01–0.02 at 1e4, Fig. 4); depth affects only convergence speed. NoPeek with small α is
  evaded; data is 32×32×3. Implemented as ts.privacy.FSHA (2026-10-18).
- **Label Leakage and Protection in Two-party Split Learning** (Li, Sun, Yang, Gao, Zhang, Xie, Smith, Wang) — 2022, ICLR — https://arxiv.org/abs/2102.08504 — norm and direction attacks recover labels from the returned gradients; defense: Marvell noise. The attack heuristics target the binary/imbalanced case. Full text read (2026-10-18, §3.3, §5): norm score r_n(g)=‖g‖₂; cosine score r_d(g)=cos(g, g+) with g+ "a positive class clean gradient" chosen at random per batch (oracle); leak AUC computed per batch; a run is summarized by the 95 % quantile over batches. ISIC (6 conv, cut after conv 4, g ∈ R^1600), Criteo, Avazu: no_noise leak AUC ≈ 1 throughout training (norm > 0.9 in Fig. 2c); Marvell s = 4 → ≈ 0.5; isotropic Gaussian ("iso") is the baseline.
- Not found: any SL/pipeline paper where a party fakes or induces a failure to gain weights or a shallower view. Searches: "split learning adversarial party exploits recovery re-partition", "fault tolerance mechanism exploited privacy attack", "client dropout manipulation attack". The dropout-attack literature targets SecAgg (privacy) or FL accuracy (DROPFL). `[negative search result, not proof of absence]`

## Model extraction / functionality stealing in SL (retrieved 2026-09-29, for G5 "what the recovery node keeps")
- **UnSplit** (Erdogan, Kupcu, Cicek) — 2022, WPES — https://arxiv.org/abs/2108.09033 — replaces the earlier
  `[abstract-only]` eprint entry above. An honest-but-curious server that knows only the client architecture recovers
  inputs and "a functionally similar model to the client model, without being detected" (coordinate descent on a
  randomly initialised copy). It also infers labels perfectly when only the output layer is hidden.
  Full text read (2026-10-18, §3.2, §4.1, Table 2): x~ = argmin MSE(f1~(x~), z) + λ TV(x~) alternated with the clone
  θ1~ = argmin MSE (coordinate descent); Adam lr 1e-3; λ 0.1 (splits 1–3), 1 (deeper); TV summed over pixels; 5 sets of
  10 images. MNIST MSE after training: 0.048 (depth 1), 0.076 (2), 0.065 (3), 0.124 (4), clone accuracy 97.45 % → 63 %
  with depth (reference 98 %). Pixel scale not stated `[unverified: probably [0, 1]]`. Implemented as ts.privacy.UnSplit.
- **PCAT: Functionality and Data Stealing from SL by Pseudo-Client Attack** (Gao, Zhang) — 2023, USENIX Security —
  https://www.usenix.org/conference/usenixsecurity23/presentation/gao `[abstract-level via search snippet + slides title]`
  — a semi-honest server with a tiny aux set (0.1–5% of the training set) trains a pseudo-client that steals the
  client's functionality, reconstructs inputs and infers labels. No knowledge of the client model is needed, and it works
  for deeper client nets. **Baseline for G5:** a *downstream* party can already extract a surrogate of everything
  upstream of it in normal operation.
- **Model Extraction Attacks on Split Federated Learning** (Li, Rakin, Chen, Yang, He, Fan, Chakrabarti) — 2023, arXiv
  `[unverified: venue; an OpenReview page exists]` — https://arxiv.org/abs/2303.08581 — malicious clients extract the
  *server-side* model by querying gradients: >90% accuracy, <2% degradation (5-layer server part, VGG-11/CIFAR-10);
  data-free when the server part is small. So extraction also works *upstream → downstream* via gradients.
- **Oops!… They Stole it Again: Attacks on Split Learning (SoK)** (Khan, Michalas) — 2025, arXiv —
  https://arxiv.org/abs/2508.10598 — covers poisoning, backdoor, model inversion, label/feature inference. **Does not
  cover** dropout/failure/recovery-triggered attacks or multi-hop chains (checked 2026-09-29).
