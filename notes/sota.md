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
- **Depth Gives a False Sense of Privacy: LLM Internal States Inversion** (Dong et al.) — 2025, USENIX Security — https://arxiv.org/abs/2507.16372 — inverts middle-layer states; 4,112-token prompt recovered at 86.88 token F1 from middle layer of Llama-3; 4 defenses insufficient.
- **Privacy Failure in Split-LLM Training: The Returned Gradient Nullifies the Decoys** (Politis, Pappas) — 2026, arXiv — https://arxiv.org/abs/2609.04382 — decoy rows get zero gradient → 4096/4096 real rows identified; forward-only privacy checks miss gradient channel. Vanilla 2-node.
- **UnSplit** — 2022, WPES — https://eprint.iacr.org/2021/1074 `[abstract-only]` — honest-but-curious server recovers inputs + client model knowing only architecture.
