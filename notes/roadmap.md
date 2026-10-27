# TorchSlicer roadmap (created 2026-10-14, updated 2026-10-15, M6 done)

Goal: become the de facto framework for split learning (SL) experiments, the way Flower is for FL, and
publish it (target: MLSys, M15).
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

- M1 (branch `feat/m1-split-anything`, forked from M0's tip, not merged/pushed): **done** =
  named cuts (`cuts=`, dotted names trace inside, an expanded module name = its output), `expand=` for
  finer granularity, `params`/`flops`/`profiled` strategies (optimal min-max over `valid_cuts()`;
  measuring ones run on a copy, so model state and RNG are untouched), `ts.trace()`/`summary()`,
  `TraceError` pointing to `pack=` instead of the silent fallback, and tutorials
  `docs/tutorials/{custom_strategy,pack}.md` (executed by `tests/test_docs.py`). Auditing the tracer found 3 silent
  correctness bugs (dropped functional ops, silent sequential fallback, lax validation) and one crash
  (a slice starting with an in-place op), all fixed (`notes/validation.md` round 2).
  Carried: recovery re-slices with its own uniform split, not the run's strategy (fixed in M2); `uniform` ignores
  `valid_cuts()` (kept so existing partitions don't move).

- M2 (branch `feat/m2-p2p`, forked from M1's tip, not merged/pushed): **done** =
  P2P in the library: `ts.P2PExecutor` (a `DistributedExecutor` whose worker 0 runs in-process) and
  `ts.run_p2p(model_fn, data_fn, **slice_kwargs)`. `ts.start_worker()` starts a worker without blocking.
  The P2P/LM examples are now model + data + one call, so the duplicated LM drivers are gone, and
  the LM examples gained tied weights and every split option. Failure detection is always on: a dead
  worker raises `WorkerFailureError` instead of hanging the run, in both topologies. The tutorial
  `docs/tutorials/p2p.md` covers the 4 extension points, with a worked example that replaces discovery
  (`FileDiscovery`). **Capability block:** workers report device, memory, `compute_gflops`
  (benchmark or declared) and caps/`first_slice` (at registration, and via the `describe` RPC);
  the executor measures RTT/bandwidth (`echo`) and, optionally, neighbour links (`probe_link`);
  planning is deferred (`SlicedModel.plan(nodes)`, called after discovery and in recovery);
  `balanced_cuts` handles speeds, memory caps and per-cut link time. Bugs fixed: validation.md
  issues 9–11 (heartbeats wiped profiling stats, a dead worker hung the run, `uniform` made fewer
  partitions than asked). Version 0.4.0.
  Carried: see the Handoff.

- M3 (branch `feat/m3-run-anywhere`, forked from M2's tip, not merged/pushed): **done** =
  one script + one config on every target. `ts.run(model_fn, data_fn, **slice_kwargs)` is the node
  entry point: `deploy.topology` (local/centralized/p2p) and the node's role (`ROLE`:
  all/coordinator/worker/driver/follower) pick what it does. `ROLE=all` (the default) runs every node
  in one process, with workers on loopback over real gRPC/TCP. `python -m torchslicer launch SCRIPT -c
  CONFIG --target local|processes|compose|k8s|baremetal` derives the node list once from the config
  (`deploy.plan_nodes`) and runs it (local, processes) or renders it (compose; k8s = ConfigMap +
  Indexed Job of workers behind a headless Service + coordinator/driver Job; baremetal = one script
  per host). One guide per environment in `docs/deploy/`; example `examples/run_anywhere/` +
  `experiments/run_anywhere_{local,centralized,p2p}.yaml`; `make launch`. On CPU, local, in-process,
  OS processes (both topologies, both transports, both discovery backends) and the bare-metal scripts
  (run on 127.0.0.1) all give exactly the unsplit losses. Bugs fixed: validation.md issues 12–15
  (`start_worker` ignored `network.device`; local ignored it too; with static discovery the plan
  depended on which nodes were up first; the documented config priority was backwards). Version 0.5.0.
  **Deviation from the plan:** k8s workers are an Indexed Job, not a StatefulSet. Same stable ordered
  DNS names, but a finite run completes instead of its workers being restarted forever (rationale in
  `docs/deploy/k8s.md`).

- M4 (branch `feat/m4-sl-settings`, forked from M3's tip, not merged/pushed): **done** =
  SL settings in `torchslicer/sl/`. Tutorial: `docs/tutorials/sl_settings.md`. Example:
  `examples/sl/train.py` with `experiments/sl_{u_shaped_p2p,splitfed_local}.yaml`.
  - **Roles:** the *data owner* is the process holding the data (local process, coordinator,
    driver). `sl.setting` decides what it hosts in-process: `pipeline` hosts none (centralized
    default), `vanilla` hosts slice 0 (P2P default), `u_shaped` hosts slice 0 and the last slice,
    so labels never leave either. The P2P driver's in-process hosting moved into
    `DistributedExecutor`, so a centralized coordinator can be the data owner too. The tail slice
    is served on `sl.tail_port`.
  - **`CutHook`:** encode/decode for activations and gradients, one method per side of a cut.
    Built-ins: Float16/BFloat16, Quantize 1–8 bits, TopK, GaussianNoise, Recorder (attack
    observer), Chain. Payloads are packed so real bytes go on the wire. Config `sl.cut_hooks`.
  - **Metrics in every history entry:** bytes per cut and direction, raw inputs/labels sent,
    bytes crossing the data owner's boundary, compute per slice (client/server), and simulated
    link time (`sl.link_*`). Local and distributed counts are identical.
  - **`ts.ClientSession`:** `sequential` (handoff copy/keep/callable) or `parallel`
    (`on_round_end`, e.g. FedAvg in user code; the core never averages). It swaps client-slice
    states between batches.

  Checked: every setting, hook and session matches local (and unsplit without lossy hooks) in
  both topologies and both transports, including real OS processes. Bugs fixed: validation.md
  issues 16–17 (uint8 inputs crashed centralized workers; the last worker leaked late labels).
  Version 0.6.0. **Deviation from the plan:** sessions simulate all clients in the data owner's
  process (see Carried).

- M5 (branch `feat/m5-offload`, forked from M4's tip, pushed): **implemented, benchmarks partly
  done**. `torchslicer/offload/`: `ts.OffloadExecutor` (cpu or disk tier, GPU or CPU optimizer
  step, chosen with `auto`), the `memory` strategy, and activation-aware `node_memory`. Survey:
  `notes/offload.md`. Bugs and measurements: validation.md round 6 (issues 18–21). Exact against
  unsplit in the matrix. 666M model under a 6 GB cap: 0.84 s/step, against 0.61 s for unsplit with
  no cap and 5.1 s for FSDP2 offload + checkpointing. Version 0.7.0.
  **Deferred, owner decision 2026-10-15** (come back after the other milestones):
  - DeepSpeed ZeRO-3/2 baselines: they exceed this container's 16 GB of RAM at 666M and OOM-killed
    it once;
  - the 1.3B disk run, larger than RAM (`benchmarks/offload/configs/gpt1b3_disk.yaml`);
  - the GPipe batch-32 config;
  - repeats (mean ± std).

- M6 (branch `feat/m6-nat`, forked from M5's tip, not merged/pushed): **done** = NAT-friendly
  transport, `torchslicer/transport/tunnel.py`; tutorial `docs/tutorials/nat.md`.
  - A node behind NAT (`transport.nat`) opens no port. It dials a `TunnelService.open` stream to
    each node that calls it, and every call to it (gradients, U-shaped tail activations, metrics,
    errors, heartbeats, tied weights, `init`) comes back on that stream.
  - A per-node `Router` routes each call: in-process, inbound tunnel, relay or direct.
    `router.channel()` replaces `grpc.insecure_channel`, so stubs and call sites are unchanged.
  - Optional relay: `python -m torchslicer relay`, `transport.relay` (`auto` = a launcher node).
    Covers NAT'd neighbours and a NAT'd coordinator with registration.
  - Checked: `tests/test_nat.py` (24 tests: settings × topologies × transports, NAT'd workers,
    relay, tied weights, failure/recovery, real OS processes). Matrix modes `p2p-nat` and
    `p2p-relay` are bit-exact on all 9 models; the full CPU matrix is 97/99 (the 2 known GPipe +
    BatchNorm cells).
  - Cost (`benchmarks/nat/`, mean ± std of 3): the tunnel is free vs direct gRPC; +27% vs direct
    raw TCP, because the reverse traffic uses gRPC; the relay is +11%.
  - Bug fixed: validation.md issue 22. Version 0.8.0.
  - **Deviation from the plan:** not only the gradient comes back in a reply. Every reverse call
    rides the client's own connection (rationale in validation.md round 7).

- M8 (branch `feat/m8-algorithm-api`, forked from M6's tip, pushed): **done** = `ts.Algorithm`
  (`torchslicer/algorithm/`), a Flower-Strategy-style extension point. Built-ins: Default
  (vanilla SplitNN, bit-exact), SplitFed V1/V2, LocalLoss; `fedavg()`; registry and
  `sl.algorithm`. Survey: `notes/algorithms.md`; tutorial `docs/tutorials/algorithm.md`;
  `tests/test_algorithm.py`.

- M9 (branch `feat/m9-fast-transport`, forked from M8's tip, pushed): **done, NCCL unvalidated** =
  `transport.tensor: tcp` with pinned-memory async D2H, plus new `ipc` (CUDA IPC, falls back to
  TCP for cross-machine peers and CPU tensors) and `nccl` (isend/irecv on a process group).
  Tutorial `docs/tutorials/transport.md`; benchmarks `benchmarks/transport/`.
  - Matrix modes `central-ipc` and `p2p-ipc`: 36/36 OK (9 models, CPU and CUDA). A probe
    confirmed the CUDA runs really used IPC. `central-nccl`/`p2p-nccl` (opt-in
    `MATRIX_NCCL=1`) are bit-exact on CPU, where they run on gloo.
  - Benchmark (redone 2026-10-17 after three measurement errors; experiments.md):
    - IPC: 4.6× over TCP and 68× over gRPC at 64 MB cuts; its transfer is 10× cheaper than
      TCP's.
    - Pinned memory: 1.8× over the old TCP path at 32–64 MB, no difference ≤ 8 MB.
    - For sub-MB cuts the transport barely matters.
  - NCCL on one GPU now fails at init with the reason, instead of hanging. Cut hooks work
    with IPC and NCCL; payloads are packed on the GPU.
  - **Real NCCL validated 2026-10-20** across two containers on the same host (one A100 each,
    NCCL socket transport): two bugs fixed (validation #38–39: `device_id` without index; a
    deadlock at the first transfer); cross-container matrix 54/54 OK; benchmark: NCCL ≈ pinned
    TCP up to 8 MB, 0.75–0.82× at 32–64 MB (sockets on both; NCCL's edge needs RDMA/NVLink).
    Still unmeasured: NCCL over NVLink/P2P (two GPUs in one container) and over RDMA.
  - **Open: thread oversubscription.** torch uses the host's core count, not the container's
    CPU quota, which made gRPC/TCP 4–5× slower here. The benchmark sets
    `OMP_NUM_THREADS`; the framework doesn't yet.
  - gRPC above 32 MB (issue 26): the slowdown was page faults on fresh ≥32 MB buffers,
    repeated over its ~9 copies per transfer. `deserialize_tensor` now copies once: 10–12 %.
    `MALLOC_MMAP_MAX_=0 MALLOC_ARENA_MAX=1` gives 4× (documented in the tutorial, not set by
    the framework). gRPC stays ~3.6× slower than TCP for big cuts; a streaming RPC would fix
    that (not planned).
  - **Who gains:** only runs with slices in separate processes: `ipc` on one machine (2–8×
    over the old TCP), pinned `tcp` across machines for cuts ≥32 MB. The local executor
    (simulator) has no transport and is unchanged. Real networks are unmeasured (loopback only).
  - Bugs and details: validation.md "M9" (issues 23–26). Version 0.9.0.

- M11 (branch `feat/m11-data-scale`, forked from M9's tip, pushed): **done** (2026-10-18) =
  `ts.data`: partitioners (iid, dirichlet, shards, quantity, natural) + feature shift; datasets
  (mnist, fashion_mnist, cifar10/100, shakespeare, synthetic, npz with groups); a `data:` config
  section, so `ts.run(model_fn)` runs a whole experiment from YAML; evaluation (`test_loss`,
  `test_accuracy`, per-token, per-client) on the local executor. Many clients: no-copy switches,
  `sl.client_states: cpu|gpu|auto` with a memory check before training, client sampling
  (`select_clients`). 1000 clients × 2.1M-param client slices: 1.5 s/round on the GPU store.
  Simulated client profiles (`sl.client_profiles`), and the link model calibrated against real
  runs (`benchmarks/transport/fit_link.py`): simulated time within 0–12 % of real when compute
  or cut size dominates, under-predicting tiny models by up to ~50 %.
  - Bugs fixed: SplitFed's last client missed the aggregate and every client inherited the
    previous one's momentum (since M8); the distributed executor silently ignored most
    `ts.Algorithm` hooks (now refused); evaluation shifted the dropout RNG (fixed before release).
  - Later (not capabilities): batched client execution, a disk tier for client states, a real
    FEMNIST/LEAF download, a tabular dataset.
  - Open: offload CUDA tests fail under GPU load (race, M5); IPC test intermittent in full runs.
  - Details: validation.md "M11" (issues 27–29). Tutorial `docs/tutorials/data.md`. Version 0.11.0.

- M12 (branch `feat/m12-privacy`, forked from `fix/post-m11`): **done** (steps 1–4, 2026-10-18) =
  `ts.privacy`: attacks observe a cut on the receiving side (after the hooks) and are scored
  against the true batch (simulator only): label inference from gradients (norm / direction,
  Li et al. leak AUC, mean + q95), black-box input inversion (decoder trained online on aux
  queries; PSNR/SSIM), distance correlation; defence `NoPeek` (a `Defence`, composes with any
  `Algorithm`) plus the existing noise/quantize/top-k hooks. `leak_*` in every epoch entry;
  config `privacy.*` or `train(privacy=)`; distributed refuses. Bit-exact losses with attacks
  on; NoPeek equals a reference loop. MNIST (3 seeds): direction AUC 1.0 → 0.50 with gradient
  noise; PSNR 42.6 / 26.4 / 15.4 dB at cuts 1/4/8; NoPeek α=10 at cut 4: 26.4 → 18.6 dB for
  −4.5 pts accuracy. Tutorial `docs/tutorials/privacy.md`; benchmark `benchmarks/privacy/`.
  Version 0.12.0. **Step 4** (same day): `FSHA` (a malicious server: `forge(obs)` replaces the
  returned gradient), `UnSplit` (architecture-only: re-initialized client copy),
  `EmbeddingRecovery` (Dong et al. ER on LLM hidden states, white-box client copy),
  `ClipNoise` hook + `DPAccountant` (GDP, per-sample rows, scope stated). 359 tests. Results:
  FSHA partly reproduced at our budget (cut2 SSIM 0.55 vs 0.16 honest), UnSplit negative,
  GPT-2 ER 93 → 58 % tokens from cut 1 to 9, label DP on CIFAR-10 binary: attacks at chance from
  z = 0.3 for ~1 pt accuracy, ε = 5.2 at z = 3. Open items in `notes/privacy.md`.

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
  - **Self-registration with capabilities + capability-aware planning** (added 2026-10-15, owner request), shared by
    centralized and P2P. M4 roles and recovery build on it:
    1. *Capability report:* extend `NodeInfo` from declared fields (device, free memory, tags) to measured ones:
       compute speed (short micro-benchmark), bandwidth/RTT to the coordinator and neighbouring peers, and optional
       constraints (max memory, "must hold slice 0").
    2. *Deferred planning:* `ts.slice()` stores graph + strategy; the executor computes partitions *after* discovery and
       passes the nodes to the strategy (`split(graph, n, nodes=...)`; old strategies ignore `nodes`).
    3. *Heterogeneous cost strategies:* min-max of `cost / speed_j` over ordered devices (same exact DP), per-device
       memory caps, and optionally the activation bytes at each cut divided by that link's bandwidth. Recovery re-plans
       for the survivors with their real capabilities instead of the blind uniform split.
    - Test: on simulated heterogeneous nodes, the plan follows their speeds and memory caps, and training still
      matches unsplit. Research link: gap G4 in `notes/gaps.md`.
- **M3, run anywhere**: one config gives local / compose / bare metal / k8s (StatefulSet = stable ordered DNS names,
  Job for the coordinator/driver); one deployment guide page per environment.
  Compose and k8s are only rendered and checked for structure in M3 (no Docker/kubectl in this container);
  running them is deferred to M13.
- **M4, SL settings**: roles (the data owner hosts stage 0 so raw data stays local; label owner; U-shaped); `CutHook`
  (compression, quantization, noise, attack observers); built-in bytes-per-cut / client-compute / simulated-link metrics;
  multi-client sessions (sequential/parallel) with a client-weight handoff hook, no averaging in the core.
- **M5, out-of-core training on one node ("virtual memory" for slices)** (added 2026-10-15, owner request; a proof
  of concept): train a model whose weights + optimizer state + activations do not fit in GPU memory (and, as a
  second tier, not in RAM) by keeping slices in a slower tier and paging one in at a time. Slices are already
  self-contained (own params, optimizer, detached cut input `SplitLayer.x`), so the plan is: forward slice by slice
  without a graph, keeping only boundary activations; backward in reverse order, paging slice *i* in, recomputing its
  forward from the stored boundary input, backward, optimizer step, paging out. The planner picks the cuts with the
  existing memory caps (`balanced_cuts`, cap = GPU budget).
  - **Step 1, survey first (before any code):** search the current technologies that solve this problem and record
    them in `notes/offload.md`: one entry per system/paper (link/DOI, year, which tiers GPU/CPU/NVMe, what is
    offloaded (params, grads, optimizer state, activations), how transfers overlap compute, reported throughput
    vs. bound, license, whether it can be called from TorchSlicer). Candidates to check, not yet verified: DeepSpeed
    ZeRO-Offload / ZeRO-Infinity, FlexGen, HF accelerate `device_map`/disk offload, PyTorch FSDP CPU offload,
    `torch.autograd.graph.save_on_cpu`, `torch.utils.checkpoint`. Literature rules apply (only what was retrieved,
    always a link). End with a short decision: what to reuse, what to write, and why.
  - **Constraint, efficiency:** it must be as efficient as the hardware allows, not a naive loop. At least: pinned host
    buffers and `non_blocking` copies; prefetch slice *i±1* on a side CUDA stream while slice *i* computes;
    double-buffered GPU slots allocated once (no allocator churn); optimizer step where it is cheaper (GPU vs. CPU,
    decided by measurement); GPipe micro-batches to amortize each page-in over several micro-batches. Report achieved
    throughput as a fraction of the transfer bound (bytes moved per step / measured PCIe or disk bandwidth) and
    explain any gap.
  - **Constraint, reuse:** reuse what exists instead of rewriting it: TorchSlicer's own pieces (SplitLayer,
    LocalExecutor + GPipe path, planner memory caps, profiler, equivalence harness) and, where the survey shows a
    mature primitive (e.g. NVMe async I/O, checkpoint RNG replay), call it, as an optional extra. PyTorch stays an
    undeclared dependency; nothing heavy becomes mandatory.
  - **Correctness:** losses equal unsplit PyTorch (same 1e-5 matrix rule). Recompute must replay dropout RNG and must
    not update BatchNorm running stats twice. The memory estimate must include activations (open since M2), or the
    planner cuts slices that don't fit.
  - **Test:** cap GPU memory (`torch.cuda.set_per_process_memory_fraction`) so a model OOMs without offload; with
    offload it trains, matches the reference, and its peak memory is reported. Compare speed against the best
    reused/existing tool from the survey under the same model, batch and cap (baseline, same setting).
  - **Shape:** a local-topology option (e.g. `pipeline.offload: none|cpu|disk`); later, the same mechanism lets a
    distributed worker hold more than its GPU fits (link to M4: a memory-poor data owner can still host slice 0).
- **M6, NAT-friendly transport**: the gradient comes back in the reply (clients need only outbound connections); optional relay.
- **M7, security: encrypted, authenticated transport** (added 2026-10-15, owner request; **deferred to before M13**, 2026-10-16 owner decision — research framework on trusted clusters; closes M6's open security items). Today every channel is plaintext and unauthenticated: gRPC (`insecure_channel`), raw TCP tensors,
  tunnels and the relay. Anyone who can reach a node can call any RPC, and `init` deserializes the slice, cut hooks
  and optimizer/criterion params with `torch.load(weights_only=False)` (`executors/worker.py`), so **one
  unauthenticated `init` call is remote code execution** on a worker. The relay trusts every HELLO, so a node can
  claim another node's address and receive its calls, and it dials any target address directly (an open proxy).
  - **Step 1, threat model first (before any code)**, in `notes/security.md`.
    - **Assets:** raw data, labels, cut tensors, weights, the worker hosts themselves.
    - **Attackers:** a network observer; an outsider who can reach a port; a malicious relay; a malicious or
      curious participant node.
    - **What this milestone defends:** confidentiality, integrity and authentication of the transport, and
      authorization of RPCs.
    - **What it doesn't:** what a curious *participant* learns from cut tensors is SL privacy (noise, attacks,
      defences). That is M11 (privacy suite), not transport security. Say so explicitly.
    - **Survey:** how comparable systems handle this (gRPC TLS/mTLS, Flower's SSL + node authentication, PyTorch
      RPC/TensorPipe, Ray TLS), with links. Literature rules apply.
  - **TLS everywhere, one config:** a `security:` section (`tls: true`, `ca`, `cert`, `key`; env vars).
    - Covers every gRPC server and channel: worker, coordinator, tail slice, relay, tunnels, registration.
    - The raw TCP tensor transport gets `ssl`-wrapped sockets.
    - The `Router` is the one place channels are made, so most of this lands there and in the servers'
      `add_insecure_port`.
    - `python -m torchslicer certs` makes a dev CA and per-node certs, so tests and `ROLE=all` stay one command.
  - **Authentication: mTLS node identity.** The certificate CN/SAN is the node's address/name.
    - The coordinator only accepts registrations, and workers only accept `init`/`shutdown`, from identities
      the run trusts.
    - The relay only routes an address to the tunnel whose certificate owns it (fixes the HELLO spoofing), and
      only forwards to known identities (no open proxy).
    - Lighter option for experiments: a shared per-run token (HMAC over call metadata), clearly documented as
      weaker.
  - **Authorization per RPC:** a small role table.
    - `init`, `shutdown`, `save_checkpoint` and `get_stats` come from the coordinator/driver.
    - `forward` and `backward` come from the neighbours named in the slice config.
    - `tied_*` comes from the tied peers.
    - Everything else is rejected with `PERMISSION_DENIED`.
  - **Safe deserialization:** stop running arbitrary pickles from the network.
    - Option A: `weights_only=True` plus an allow-list of module classes (`torch.serialization.add_safe_globals`).
    - Option B: architecture by name/import path plus state dicts.
    - Hooks and optimizer params become plain data plus a registry name.
    - Keep `weights_only=False` only behind an explicit `security.trust_pickles` for trusted clusters. Measure
      what breaks (HF/peft modules, `pack=` stages) before choosing.
  - **Optional end-to-end encryption through the relay.** TLS is hop by hop, so the relay sees plaintext.
    Optionally encrypt tunnel frame payloads per node pair (AEAD, keys bound to the mTLS identities), so the relay
    only routes ciphertext. This is an optional extra (`cryptography`), never mandatory.
  - **Constraints:**
    - All of it is optional and off by default. Today's plaintext mode keeps working for local experiments.
    - No new mandatory dependency: gRPC and `ssl` are built in.
    - A misconfiguration fails at start with a clear message, never silently in plaintext.
  - **Correctness and tests:**
    - TLS/mTLS runs match plaintext losses and bytes exactly. Add equivalence matrix modes `p2p-tls`,
      `p2p-relay-tls`.
    - Negative tests:
      - a wrong or absent cert is rejected;
      - `init` from a non-coordinator gets `PERMISSION_DENIED`;
      - `forward` from a non-neighbour is rejected;
      - a spoofed HELLO at the relay is refused;
      - a malicious pickle in `SliceConfig` is not executed in safe mode.
  - **Cost:** TLS/mTLS/e2e overhead vs plaintext with `benchmarks/nat/`-style runs (mean ± std over 3), and the
    handshake/start-up cost.
  - **Shape:** TLS covers the M13 real-environment runs. Two nodes on a real network with mTLS is the first
    real-network check of this milestone.
- **M8, algorithm API** (added 2026-10-15, owner review). It must come before the baselines, or every baseline
  hard-codes engine changes.

  **What exists today:**
  - `CutHook`: what crosses a cut;
  - `ClientSession`: `handoff` / `on_round_end`, for client-state policy and e.g. FedAvg;
  - splitters, `TrainingCallback`, and the P2P extension points (discovery, plan, scheduling, failures).

  These change *placement*, *payloads* and *client-state bookkeeping*. None of them changes the training step
  itself: forward → loss on the last slice → full backward chain → step after every batch is fixed inside the
  executors (`LocalExecutor`, `WorkerServicer`).

  **What can't be expressed today:**
  - local-loss / no-gradient-back SL;
  - per-client server copies, as in SplitFed v1;
  - server-side aggregation;
  - client sampling or straggler policies;
  - loss changes for label protection;
  - gradient manipulation at the server;
  - asynchronous updates.

  **Deliverables:**
  - **An `ts.Algorithm` interface** (Flower-Strategy-like, small) with hooks around one step, each with a
    default equal to today's behaviour, so existing runs don't change:
    - `client_step` / `server_step`: what each side computes and returns;
    - `loss`: on which slice, and auxiliary losses;
    - `backward_policy`: full chain / stop at cut *k* / local loss;
    - `aggregate`: client and server states, per round;
    - `select_clients`: per round;
    - `on_round_end`.
  - **The same algorithm object runs in the local, centralized and P2P executors.** The executors call the
    hooks, so it is written once.
  - **Port the existing features onto it:** vanilla, U-shaped, sequential SplitNN, SplitFed-style FedAvg
    (`ClientSession` becomes a thin layer over it). Then implement 2 that are impossible today, to prove it:
    - local-loss SL with a client auxiliary head and no gradient back;
    - per-client server copies with server aggregation.
  - **A registry and a tutorial**, "write an SL algorithm in < 100 lines", executed by `tests/test_docs.py`.
  - **Survey first** (`notes/algorithms.md`): the SL algorithm families and what each needs from the engine:
    vanilla, U-shaped, SplitFed v1/v2/v3, parallel SL, local-loss / asynchronous SL, CPSL, FSL, label-protection
    schemes. Literature rules apply.
  - **Test:** the default algorithm reproduces the current matrix bit for bit; each new algorithm matches a
    hand-written reference loop.
- **M9, fast tensor transport** (added 2026-10-16, owner decision after benchmarking on A100).
  Benchmarks on an A100 MIG (40 GB, loopback) showed TCP is already 1.4–3.5× faster than gRPC, but
  both are far below PCIe bandwidth because every tensor goes GPU → CPU → socket → CPU → GPU.
  Results: `benchmarks/transport/` (small 1 MB: gRPC 11 ms → TCP 3.9 ms; medium 33 MB: 28.8 → 8.2 ms;
  large 8 MB: 12.5 → 8.8 ms; xlarge 50 MB: 40.3 → 26.2 ms, ×3 seeds).

  **Three transports to add, in priority order:**

  1. **Pinned memory + async D2H/H2D** (`transport.tensor: tcp`, improvement to existing path).
     Allocate one pinned host buffer per peer connection at `init`. Use `tensor.to(pinned,
     non_blocking=True)` + a CUDA event to signal readiness before `sendall`, and
     `torch.cuda.current_stream().synchronize()` only when the buffer is needed. Overlaps GPU
     compute on the *next* slice with the copy of the *current* slice's activation. Compatible
     with the existing TCP frame format; no API change.
     - Expected gain: 1.5–2× on top of TCP, especially for large tensors where the D2H copy
       dominates. Measure with the `benchmarks/transport/bench_transport.py` script.
     - **Constraint:** the pinned buffer must be size-stable (allocate at max expected activation
       size, or realloc with a high-water mark). Never allocate per-batch.

  2. **CUDA IPC** (`transport.tensor: ipc`, new mode, same-node only).
     Export each GPU tensor as a CUDA IPC handle (a ~100-byte descriptor), send only the handle
     over the existing socket, and let the receiver reconstruct the tensor by mapping the same
     CUDA allocation — zero host copy. `torch.multiprocessing` already has the plumbing
     (`ForkingPickler`, `reduce_tensor` / `rebuild_cuda_tensor`).
     - Expected gain: 10–50× over TCP on same-node, approaching PCIe P2P bandwidth (~30 GB/s).
     - **Constraint:** only works when both workers share a CUDA device (same machine). The
       `Router` / discovery should detect cross-machine peers and fall back to TCP automatically.
       Guard with `torch.cuda.is_available()` and a clear error if IPC is requested cross-node.
     - **Lifecycle:** the sender must keep the tensor alive until the receiver signals consumption
       (a tiny ACK over the socket); otherwise the CUDA allocator can reuse the memory.

  3. **NCCL P2P** (`transport.tensor: nccl`, new mode, multi-node with RDMA NICs).
     Use `torch.distributed.isend` / `irecv` with the NCCL backend for GPUDirect RDMA:
     GPU memory → NIC → GPU memory on the peer, CPU never touched.
     - Expected gain: near wire-speed on InfiniBand / NVLink clusters.
     - **Constraint:** requires all workers to share a `ProcessGroup`. The executor initialises
       one at setup (rank = slice index, world = n_slices); the coordinator/driver participates
       as rank 0. This is architecturally heavier and may not compose cleanly with fault
       recovery (process groups are torn down on failure). Implement after IPC is working.

  **Deliverables:**
  - Pinned-memory path in `executors/worker.py` (`_TensorPeerClient`) and `benchmarks/transport/`
    results updated.
  - New `transport.tensor: ipc` mode in `config.py`, `executors/worker.py`
    (`_IpcPeerClient`, `_IpcServer`), with automatic fallback to TCP for cross-node peers.
  - New `transport.tensor: nccl` mode, gated on `torch.distributed` availability.
  - All three new modes pass the equivalence matrix at ≤ 1e-5.
  - `benchmarks/transport/bench_transport.py` extended to cover all four modes (grpc / tcp /
    ipc / nccl) across the four model sizes, with mean ± std over ≥ 3 seeds.
  - `docs/tutorials/transport.md`: when to use which mode (≤ 60 lines).

- **M10, multi-client engine** (was M9; carried from M4; moved after M13 on 2026-10-17). Clients as real nodes, not states swapped in one process.
  - **Clients on several machines.** Covers:
    - a worker-side data source (a client node owns its loader);
    - a per-batch return address on the shared server slice;
    - an RPC to transfer client state (for handoff and aggregation);
    - NAT'd clients by default (M6).
  - **True parallel clients:** concurrent batches from many clients on the server.
    - Per-client server copies, or batching across clients, chosen by the algorithm (M8).
    - Client sampling, stragglers, dropout and late joiners, with deadlines per round.
  - **Vertical SL:** a label party distinct from the data owner, and several feature parties (carried from M4).
  - **Sessions with fault tolerance:** a failed client is dropped from the round instead of stopping the run
    (carried from M4).
  - **Test:** N client processes equal the in-process `ClientSession` for the same algorithm, and losses match.
  - **Design (2026-10-20, owner-approved: "push and proceed").** Why: simulation (`ClientSession`)
    already covers accuracy studies; what is missing is a *deployment* whose clients are real
    nodes, so overhead (time per round, server contention, bytes per client, stragglers) is
    measured, not simulated. Today a distributed session keeps every client's data and slice
    in the data owner's process. Rules: additive only (local, `ClientSession`, today's
    distributed runs and their defaults unchanged); same experiment in both (same `model_fn`,
    `Algorithm`, `sl.clients`); a deterministic schedule must give the `ClientSession` losses.
    - **Shape:** `sl.client_nodes: true` (centralized topology). Coordinator: rounds, schedule,
      `select_clients`, aggregation; holds no data and no slice. Server workers: today's chain,
      slices 1..n-1. Client nodes: a worker hosting slice 0 plus its own loader (`ts.run`
      `ROLE=client`, `CLIENT_INDEX`). The coordinator ships slice 0 to every client, so all
      start from the same weights. Executor: `ClientNodesExecutor(DistributedExecutor)`; on the
      coordinator, `ts.RemoteSession(n_clients, mode, handoff)` stands in for the loaders.
    - **Per-batch return address:** slice 1 gets every client's address (`SliceConfig.prev_workers`);
      the client of a batch is `(batch_id // n_micro) % n_clients` (the coordinator picks the
      ids), so gRPC / TCP / IPC frames are unchanged. NCCL refused (its world is fixed at init).
    - **A batch:** coordinator → `client_step(batch_id)` → the client takes its next batch, sends
      the labels to the last slice and the activation to slice 1; the gradient comes back to it;
      it reports `batch_done`. Clients only dial out (NAT works through M6 tunnels unchanged).
    - **State RPCs** `get_state` / `set_state` (state dicts, `weights_only=True`): sequential
      handoff (copy = the previous client's state to the next), `on_round_end` / `aggregate`
      over the round's participants, as `algorithm_round_end` does in-process.
    - **Steps:** (1) worker side + executor, vanilla, sequential and parallel, = `ClientSession`
      on the local executor; real OS processes; across the two containers. (2) `ts.run` /
      `launch` roles. (3) SplitFed V1 (per-client server copies on the server workers). (4)
      failures: a dead client leaves the round, late joiners, `round_deadline`. (5) concurrent
      schedule (many clients in flight; the realistic overhead mode, not loss-equal by design)
      + overhead benchmark. Later: U-shaped clients, vertical SL, a multi-worker server under
      concurrency.
    - **Status 2026-10-20:** steps 1–2 done (555bd1a): `ts.ClientNodesExecutor`, `ts.RemoteSession`,
      worker RPCs, `ROLE=client`, launcher nodes; = `ClientSession` in every schedule (tests,
      OS processes, and across the two containers). Step 3 done: SplitFed V1 (per-client server
      copies on the server workers) = in-process V1. Step 4a done: a dead client leaves the run; round deadline. Step 4b done: late joiners and
      rejoins (`sl.min_clients`, `set_peers`). Step 4c done: server-side recovery
      (`fault_tolerance.enabled`; not with V1). Step 5 done 2026-10-22: `sl.client_schedule: concurrent` (`client_round`, `sl.max_concurrent_clients`; multi-slice server gated to one batch at a time; = the in-process session in the server's order), 4.1x per epoch on 8 Orin Nano clients (validation.md). Not yet with it: V1, recovery, round deadline. Next: test accuracy under concurrency over more epochs; then U-shaped clients, vertical SL.
- **M11, data and scale simulation** (was M10):
  - **Federated partitioners:** IID; Dirichlet(α) label skew; quantity skew; per-client feature shift.
  - **Standard datasets:** MNIST, CIFAR-10/100, FEMNIST-like, a text/LLM fine-tuning set, tabular for vertical SL.
    Seeded splits saved with the run, so results are comparable across papers.
  - **Many-client simulation on one machine:** hundreds or thousands of virtual clients, with client states
    off-device (reuse M5 paging) and a scheduler over a GPU pool.
  - **Heterogeneity profiles:** compute speed and link bandwidth/latency per client, from the M2 capability model
    and the M4 link model, as config.
  - **Batched client execution** (added 2026-10-17): today `ClientSession` runs clients one at a
    time, swapping states. Run many clients' slices in one batched forward/backward (e.g.
    `torch.func` + `vmap` over stacked client states) where the algorithm allows, so a round
    costs about one big batch instead of N small ones. Must match the sequential path's losses.
  - **Calibrated simulated time** (added 2026-10-17): check `sim_*_comm_s` against the M9 transport
    measurements (`benchmarks/transport/`), so simulated wall-clock can be trusted.
  - **Test:** partition statistics match their definitions; a 1000-client run fits in this container's RAM.
- **M12, privacy: attacks, defences, leakage metrics** (was M11). SL papers are judged on this, and the core has only
  `Recorder`.
  - **Attacks:**
    - label inference from returned gradients;
    - UnSplit / model inversion from cut activations;
    - feature-space hijacking (malicious server);
    - for LLMs, internal-state inversion (see `notes/sota.md`).
  - **Defences:** Gaussian/DP noise at cuts (with an accountant), NoPeek-style decorrelation loss, label
    protection, and quantization/top-k as side effects.
  - **Leakage metrics** in every history entry when an attack is configured: reconstruction PSNR/SSIM, label
    accuracy, distance correlation.
  - Attacks are `CutHook` observers or `Algorithm`s (M8), so they run in every topology.
  - **Test:** each attack reproduces its paper's qualitative result on a small setting, and each defence lowers
    it measurably. Negative results are recorded as such.
- **M13, baselines + Flower interop** (was M12).
  - **A `baselines/` template:** config, seeds, expected numbers with tolerance, a CI smoke test, mean ± std
    over ≥3 seeds.
  - **Seed it with the famous papers:**
    - SplitNN (vanilla);
    - U-shaped;
    - SplitFed v1 and v2;
    - one local-loss / asynchronous SL;
    - one compression paper;
    - one attack + defence pair (from M12);
    - one split-LLM / LoRA fine-tuning paper.
  - **Each reproduces the published number within the stated tolerance.** If it doesn't, record it as such,
    with the reason.
  - **Flower interop:** a TorchSlicer client usable as a Flower client, and Flower strategies for the
    SplitFed aggregation. This puts SL where the FL community already works; SplitBud (`notes/sota.md`) shows
    the demand.
  - **A contributor guide** for adding a baseline (cf. Flower Baselines).
  - **Status (2026-10-19, branch `feat/m13-baselines`, part 1 committed):** template + runner + CI smoke
    (`baselines/`, `docs/baselines.md`); SLPerf MNIST (vanilla, U-shaped, SplitFed V2) **not reproduced**
    (authors' SLFrame doesn't reproduce its own Table 2 either; on SLFrame's exact split TorchSlicer = SLFrame:
    vanilla 0.959 both, SplitFed 0.89 vs 0.88–0.93); SplitFed FMNIST (SL, V1, V2) reproduced ±0.01; label
    leakage + ClipNoise reproduced; Flower strategies as SplitFed aggregation (`ts.interop.flower`), FedAvg =
    weighted SplitFed within 1e-6; FedAvg now sample-weighted (validation #33). Part 2 (6a8c25d, db27ad3):
    SplitLoRA GPT2-S E2E **not reproduced** (BLEU 0.566 vs 0.694, 1 seed; split = centralized), plus
    `training.scheduler`, `ts.data.register_dataset`, frozen ties allowed in sessions. Owner 2026-10-19: stop
    spending time on it. Part 3 (7add05e, dc9d48d, 2026-10-19): `ts.CircularConv` (C3-SL hook) + C3-SL
    VGG-16/CIFAR-10 baseline, **reproduced** (all 5 runs within 0.01, 3 seeds; e60a31d); `ts.SplitServer`/`ts.SplitClient` + Flower
    `split_client_app` (TorchSlicer clients in Flower's engine, 1 node = SplitFedAlgorithm exactly);
    fedavg single-client optimizer fix (#37); version 0.13.0. **Skipped (owner, 2026-10-19):**
    local-loss/async. Contributor guide checked. **Merged to main** 2026-10-20 (PR #5, 665a519; CI green after the py3.10 Flower and transformers fixes, 652b296). **M13 done.**
- **M14, real environment and testbed** (was M13; deferred from M3 on 2026-10-15, owner decision).
  - **Containers.** This container has no Docker or kubectl, so this runs in a fresh clone on a machine that
    has them:
    - rebuild the image (`make build-cpu`/`build-gpu`);
    - for both topologies, run `launch --target compose` and `--target k8s` (kind/minikube is enough) on
      `examples/run_anywhere/`;
    - check that the losses equal the `processes` target's.
  - **Bare metal** across at least two real hosts.
  - **A small physical testbed:** a GPU server plus edge clients (e.g. Jetson / Raspberry Pi) over Wi-Fi and a
    real NAT, with M7 TLS on. Measure on it:
    - real bytes and real time per round;
    - client energy if feasible;
    - NAT/relay cost over a WAN;
    - recovery from a real node loss.
  - Record every bug in `notes/validation.md` as usual.
- **M15, release readiness** (was M14; owner decisions marked; some items can happen at any time, and the owner
  decisions should be settled before M13 invites contributors):
  - **Merge** the milestone branches into `main` in order, and tag a 1.0 release candidate (owner).
  - **License:** Apache-2.0 is the norm for adoption; GPL-3.0 deters industry users (owner).
  - **Stable public API:** a list of public names, semantic versioning, and a deprecation policy.
  - **Docs site:** tutorials, API reference, deployment guides, a gallery of examples.
  - **CI:** the equivalence matrix as an integration test (a small subset per PR, the full run nightly; carried
    since M0), plus GPU CI if available.
  - **Packaging:** PyPI wheels, a Docker image on a registry, Zenodo DOI per release.
  - **Community:** a contribution guide, issue templates, a discussion channel, a simple governance note.
- **M16, the paper (target: MLSys)** (was M15). The evaluation is designed up front, and each earlier milestone records
  its numbers in `notes/validation.md` as it goes.
  - **Claims to back with measurements:**
    1. *Generality.* Any `nn.Module` (CNNs, HF LLMs, LoRA, MoE via `pack=`) is split automatically into chains
       of any length, with no manual partitioning, vs SplitBud/SLPerf, where the split is manual.
    2. *Correctness.* Training is bit-identical or within 1e-5 of unsplit PyTorch in every mode (the
       equivalence matrix), which no other SL framework shows.
    3. *Expressiveness.* The M13 baselines take N lines each with `ts.Algorithm`, compared with their original
       code bases and with SplitBud/SLPerf.
    4. *Systems cost.* Overhead vs plain PyTorch; scaling with clients and with chain length;
       capability-aware planning vs uniform on heterogeneous nodes; NAT/relay and TLS cost; recovery time
       after a node loss; out-of-core training vs FSDP/DeepSpeed offload.
    5. *Reality.* The same experiment in simulation and on the M13 testbed, and how far they differ.
       **Owner decision 2026-10-22:** the simulator reports only hardware-independent results
       (accuracy/loss, bytes and messages, memory, FLOPs); time, CPU and energy are measured on the
       testbed only, not simulated. So this claim compares *those* metrics (losses are already
       identical on the testbed, validation.md M14) and reports testbed time/energy as measured.
       Why: on the Jetsons DVFS alone changes a step's time 2-7x (validation.md, jetson_clocks A/B).
  - **Related work:** SplitBud, SLPerf, Flower, FedML, PySyft, pipeline-parallel systems (GPipe, Oobleck),
    from `notes/sota.md`, only papers actually retrieved.
  - **Artifact:** packaged for artifact evaluation, reproducing every figure from configs + seeds.
  - **Early preview** (optional): a short workshop paper (e.g. EuroMLSys, where SplitBud appeared) once M8–M9
    land, to stake the claim. Check the current calls and deadlines; none are recorded here.
- **M17, compiled slices and a `torch.export` tracer** (added 2026-10-17; not yet placed in the milestone order,
  owner to decide). `torch.fx` is used only to build the graph at slice time (`core/model_graph.py`); slices
  then run as eager PyTorch, and nothing calls `torch.compile` today. Its costs are model coverage (data-dependent
  control flow and dynamic shapes fail, hence `pack=` for HF LLMs/MoE) and an untyped graph (no shapes or costs),
  not speed. A JAX backend was considered and rejected: SL time is mostly communication, which XLA doesn't
  touch; the per-slice optimizer, recovery, `ClientSession` and offload state are imperative; and SL/FL users
  are on PyTorch (adoption goal).
  - **Per-slice `torch.compile`** (opt-in, e.g. `training.compile: false|true|<mode>`, plus env override).
    Compile each `SplitLayer`'s module; optionally CUDA graphs for fixed-shape slices. Must interact correctly
    with bf16 autocast, GPipe micro-batches, offload paging (`tensor.data` swaps into slots will
    invalidate guards: likely unsupported there, fail clearly), `ClientSession` state swaps, and recovery
    re-init. Compile cost is paid once per worker, so record it separately from step time.
  - **`torch.export` tracing backend** (e.g. `ts.slice(..., tracer="fx"|"export"|"auto")`). Build `ModelGraph`
    from the exported graph, keeping symbolic shapes on `LayerNode`s so `flops`/`params` costs and
    `valid_cuts()` can use them without profiling. `auto` tries export, then fx, then points to `pack=`.
    Slices must still be picklable `nn.Module`s shipped with `torch.save` (check that exported submodules are).
  - **Communication/compute overlap** where still missing (async sends, overlapping micro-batch comm with
    compute), since that is where SL time goes.
  - **Test:** compiled and eager slices match within the equivalence-matrix tolerance in every mode; the
    export tracer gives the same partitions as fx on every model fx handles today, and traces at least one
    model that currently needs `pack=` (record which ones still don't). Benchmark eager vs compiled step time
    and compile time on the M9 transport models (mean ± std, ≥3 seeds); a slowdown is a valid result.
- **M18, multi-tensor cuts** (added 2026-10-22, owner decision; after the FGCS special-issue paper, not placed in the
  order yet). Today exactly one tensor crosses each cut (`BaseSplitter.validate()`), so U-Net-style encoder-decoder
  CNNs (skips at every level: only cuts near the ends are valid), encoder-decoder transformers (T5, Whisper, BART:
  decoder layers cross-attend to the encoder output) and models with skips from the raw input can't be split usefully.
  Decoder-only LLMs don't need it (masks/positions are rebuilt inside the stages), so the paper's work doesn't.
  - **Step 0, cheap, any time:** when `validate()` refuses a cut, name the tensors that cross it and point to the
    workaround; ship a small tested pack/unpack stage helper (cat at the sending side, split at the receiving side,
    gradients flow through both) so users don't hand-write it.
  - **The milestone:** a cut carries a tuple of tensors. Touches: boundary validation and `Partition.predecessors`;
    transport (gRPC, raw TCP frame, IPC, NCCL send/recv; `pack_payload` already packs hooked payloads into one uint8
    tensor); the `CutHook` API (per tensor or per tuple); GPipe micro-batching; offload stash/recompute; what privacy
    attacks observe at a cut; byte metrics per cut.
  - **Test:** equivalence matrix in every mode with a U-Net and a small T5 added, cut through the skips; every existing
    cell unchanged; bytes per cut equal the sum of the crossing tensors.
  - **Started 2026-10-27** (owner: "M18 + recompute"), branch `feat/m18-multi-tensor-cuts` forked from `refinement`
    (f062ed7). Design (decided in session, evidence: tiny Llama exported with `torch.export`, 4 tensors cross every
    block boundary: hidden state, causal mask, RoPE cos/sin; only the hidden state needs sending):
    - **Graph rewrite, not a new wire contract.** `ts.slice(..., multi_tensor=True)`: every gap is a candidate; at a
      cut several values cross, `core/multicut.py::pack_cuts` inserts `cut<k>.send` (packs the values into one
      differentiable `[B, header + values]` tensor) and `cut<k>.recv<i>` nodes (unpack). Executors, workers, protos,
      transports, GPipe and offload see one tensor per cut, unchanged. Values crossing several cuts are re-packed.
      Header in-band per row (bytes as floats, exact in any float dtype); floats promoted to the widest float dtype;
      ints/bools as bytes; non-batch values (parameters, `[1, T, d]` tables, Python numbers) repeated per row.
    - **Plan / status (2026-10-27):** (1) rewrite + local exactness ✅; (2) every executor/mode ✅: GPipe, offload
      cpu/disk, centralized gRPC/TCP, P2P, `ClientSession`, client nodes, P2P recovery (fallback snaps to valid cuts),
      checkpoint restore (`packed_cuts` recorded), tied weights, eval, CUDA fp32 bitwise; bf16 equal to rounding
      (layout, validation.md M18); (3) cut hooks at a packed cut ✅ (owner: "need to support cut hooks", 2026-10-27:
      `PackedCutHook`, the hook sees the one per-sample float value in its shape, else all of them as `[B, N]`);
      privacy attacks/defences at a packed cut still **refused**; config key `planning.multi_tensor` ✅ (owner);
      (4) recompute ✅ (metadata-only values rebuilt; mask 17376 → 10464 B/cut, h alone 9216); (5) messages,
      `summary()`, `view()` ✅; (6) equivalence matrix ✅ `unet`/`encdec`/`maskdec` (`multi_tensor=True`): CPU 45/45, CUDA 42/42 OK;
      (7) tutorial `docs/tutorials/multi_tensor.md` ✅, validation.md #53 + M18 notes ✅.
    - **Model sweep (2026-10-27, owner: "did you try several models? like transformers and resnet"):** 18
      architectures, `expand="all"`, n=2 and 4, one SGD step vs unsplit + Quantize hook on every cut: all exact
      after fixing #54–#57 (validation.md); HF models still untraceable by fx (need `pack=` or M17's tracer).
    - **Follow-ups (not done):** privacy attacks/defences at packed cuts; `new_ones`/`new_zeros` (read only
      dtype/device of their tensor) rebuilt after a cut instead of sent; capability-aware planning on the base graph (today it
      re-plans among the rewritten graph's cuts); recovery fallback snapping for every graph (owner decision: turns
      an existing refusal into a working recovery); the `torch.export` tracer (M17) on top of this, then HF models
      without `pack=`; header overhead (per row) could be per tensor for large batches of small values.

## Handoff (read this first if you are a new session)
**2026-10-27: M18 (multi-tensor cuts + recompute) done on `feat/m18-multi-tensor-cuts`** (forked from `refinement`,
not pushed): see M18 status above; 532 tests (cut hooks at packed cuts, `planning.multi_tensor`, 18-model sweep
with fixes #54–#57), matrix new models CPU 45/45 (rerun after the fixes), CUDA 42/42. Open owner decisions:
recovery fallback snapping for every graph; version bump.
Last updated 2026-10-18, M12 done incl. step 4 (version 0.12.0, branch `feat/m12-privacy`, pushed); next: M13 (owner: "do both", 2026-10-18). **2026-10-20: M13 done and merged to `main` (PR #5, 665a519), version 0.13.0; see M13 Status. Next by the milestone order: M10 multi-client engine, branch `feat/m10-*` forked from `feat/m13-baselines` (`ts.SplitServer`/`ts.SplitClient` are a start on it).** NCCL validated across two containers 2026-10-20 (validation #38–39). M7 deferred.
- **Where things stand:**
  - Done, pushed to `origin`: M0 (`feat/m0-foundations`), M1 (`feat/m1-split-anything`),
    M2 (`feat/m2-p2p`), M3 (`feat/m3-run-anywhere`), M4 (`feat/m4-sl-settings`),
    M5 (`feat/m5-offload`), M6 (`feat/m6-nat`), M8 (`feat/m8-algorithm-api`),
    M9 (`feat/m9-fast-transport`). `main` is
    unchanged (merge deferred to M15 release readiness).
  - **M7 (security) deferred** (2026-10-16, owner decision): TorchSlicer is a research framework
    running on trusted clusters; plaintext mode is acceptable for experiments. M7 will be done
    just before M14 (real environment + testbed), which is the first milestone that requires
    security on a real network. The `weights_only=False` RCE on worker `init` is a known open
    risk; document it and accept it until then.
  - **M12 (privacy)**, branch `feat/m12-privacy`: done, steps 1–4 (see Status, `notes/privacy.md`,
    validation.md "M12"). Next: **M13 baselines**, branch `feat/m13-baselines` forked from
    `feat/m12-privacy`. M13's "one attack + defence pair" can use label leakage + ClipNoise (DP
    curve) or inversion + NoPeek. Papers are in `notes/sota.md` (don't re-search them).
  - **`main` is up to date** (2026-10-20): M1–M13 + `fix/post-m11` merged via PRs #3, #4, #5,
    without `notes/` (see `notes/README.md` and `notes/merge_to_main.sh`). CI: `tests.yml`
    (CPU, py3.10–3.12, on PRs and main) and `gpu-tests.yml` (self-hosted runner
    `torchslicer-a100` in this container: push to main, nightly 03:00 UTC with the CUDA matrix,
    manual; never PRs; under `.github/ci/memguard.py`). Fork PR workflows need approval
    (`all_external_contributors`). **After a container restart, the runner is down:**
    `bash ~/actions-runner/start_gpu_runner.sh` (also `stop`, `remove`). First GPU run: green.
    **This container has 16 GB RAM and may be shut down if it exceeds it:** estimate every
    scaling run first and use the watchdog in `benchmarks/clients/bench_clients.py`.
    M9's NCCL item: done 2026-10-20 across two containers (second container `10.224.0.9`, ssh
    port 2222, its own A100; `MATRIX_REMOTE`/`BENCH_REMOTE` run workers there). Same physical
    host: correctness yes, real-network overhead no (M14).
    Other open items:
    - `test_offload.py::test_cuda_matches_local` flakes (NaN), with or without the M9 changes,
      seen while another tenant held the GPU at 99 %. It may be a timing race in offload (M5).
    - Calibrate `sim_*_comm_s` against the M9 transport measurements: now part of M11.
    - Framework-side thread sizing to the cgroup CPU quota (owner decision: changes defaults).
  - **Milestone order** (updated 2026-10-17, owner decision): M9 fast transport (done) →
    M11 data and scale simulation → M12 privacy suite → M13 baselines + Flower interop →
    M10 multi-client engine → M7 security → M14 real environment and testbed →
    M15 release readiness → M16 the MLSys paper. Numbers are kept, only the order changed.
    M17 (compiled slices + `torch.export` tracer, added 2026-10-17) is not placed yet (owner);
    it should land before M16, whose "overhead vs plain PyTorch" claim it affects.
    - **Why:** the goal is adoption as the standard SL framework. Researchers mostly simulate, and
      the competitors (SLPerf: 13 variants, datasets, non-IID splits; SplitBud: Flower-based,
      multi-client) win on what a researcher meets first. So the simulator features (M11),
      the privacy suite (M12, needed by the M13 attack + defence baseline) and the reproduced
      baselines (M13) come before real multi-node clients (M10). SplitFed-style baselines
      run on the in-process `ClientSession`, so M13 doesn't need M10.
    - **Release work stays last** (owner decision 2026-10-17): the repo is already public, so
      merging into `main`, publishing to PyPI and the docs site happen in M15. The license
      can change later (owner).
    - **Never break what exists:** every milestone keeps the test suite and the equivalence
      matrix green, and doesn't change existing losses or defaults without an owner decision.
- **Read first:**
  - this file;
  - `notes/validation.md` (how correctness is checked, and every bug found so far, rounds 1–7);
  - `CLAUDE.md` (layout, commands, architecture, including P2P, planning, run-anywhere and SL
    settings);
  - `docs/tutorials/p2p.md`, `docs/tutorials/sl_settings.md`, `docs/tutorials/nat.md` and
    `docs/deploy/README.md`.
  - `notes/sota.md` and `notes/gaps.md` are for the *research* lines (`research/*` branches), not
    for framework milestones.
- **Every change must keep:**
  - `.venv/bin/python -m pytest tests/` green (359 tests at M12, about 5 min with `OMP_NUM_THREADS=4`;
    `test_sl.py::test_distributed_hooks_match_local[hook0]` flaked once under load, and
    `test_offload.py::test_cuda_matches_local` flakes on a busy GPU, see open items);
  - the equivalence matrix green: `.venv/bin/python benchmarks/equivalence/run_matrix.py` (about
    15 min on CPU, more with the M9 `*-ipc` modes; pass a model name for a quick check).
    - Set `MATRIX_DEVICE=cuda` for the GPU, and
      `MATRIX_OPT='{"name":"AdamW","params":{"lr":0.001,"weight_decay":0.01}}'` for AdamW.
    - A change to splitting must not change any loss. Every mode is compared at 1e-5 against
      full-precision losses.
- **Record as you go:** each bug found or behaviour changed goes into `notes/validation.md`, with
  the symptom, root cause, fix, cost and test, in the same commit as the fix. Note the pre-fix
  numbers *before* rerunning, because logs get overwritten. A paper on TorchSlicer will be written
  from these notes.
- **Environment:**
  - **Limits:** the container has a 16 GB RAM cgroup limit and a 4-CPU quota (`free` and `nproc`
    don't show either). Going over the RAM limit OOM-kills the whole container, so estimate a
    run's host RAM before launching it. The GPU exposes ~42 GB and has no NVML access, so a GPU
    OOM fails an NVML assert instead of raising `OutOfMemoryError`.
  - Git commits: the container has no git identity, so commit with
    `git -c user.name=... -c user.email=...`. Ask the owner for the values if your memory doesn't
    have them.
  - Downloads: HF hub and dataset downloads are blocked. Use random-init HF configs and synthetic
    data, as in `benchmarks/equivalence/tsmodels.py`. `pip download` works.
  - No Docker and no kubectl in this container: compose/k8s output can be rendered and checked,
    not run. The host has a GPU, so `DEVICE`/`network.device: auto` means CUDA here. Pin `cpu`
    when comparing against a CPU reference.
  - gRPC stubs: regenerating them needs grpcio-tools 1.64.0 (so the gencode matches the declared
    minimum). It has no wheel for the venv's Python 3.13. What worked in M2 and M4:
    1. `pip download --only-binary=:all: --python-version 3.12 --platform manylinux2014_x86_64
       grpcio-tools==1.64.0 -d wheels`;
    2. unzip the wheels into a directory;
    3. run `PYTHONPATH=<dir> /usr/bin/python3.12 -m grpc_tools.protoc ...` (the command in
       `CLAUDE.md`).

    Unchanged protos regenerate byte-identical this way.
  - The `torchslicer` console script needs a reinstall of the editable package;
    `python -m torchslicer` works without it.
  - Don't commit the owner's untracked files at the repo root (`*.prompt.md`, `prompt*.txt`,
    `results/`, `sota_check.md`) or their `.gitignore` change. Launcher output goes to
    `runs/deploy/` (ignored).
- **Facts for M5 (verified in code, 2026-10-15):**
  - `LocalExecutor` already runs one `SplitLayer` per slice, each with its own optimizer and a
    detached cut input (`SplitLayer.x`).
  - `_standard_batch` and `_gpipe_batch` run the cuts through `_cut_forward`/`_cut_backward` (hooks
    + byte counting) and time each slice (`_clock`, `CommMeter.add_compute`). An offload path must
    keep calling them, so its metrics stay comparable.
  - The memory caps in planning (`balanced_cuts`, `NodeInfo.memory_cap_mb`) count parameters, not
    activations or optimizer state. M5 needs the full estimate.
  - Compute timing synchronizes CUDA after each phase on CUDA devices (validation.md round 5,
    "Behaviour changes"). Offload overlap measurements must account for it, or make it optional.
- **Carried from M6:**
  - **TCP reverse path.** With `transport.tensor: tcp`, traffic to a NAT'd node rides the gRPC
    tunnel (+10 ms/batch in the benchmark).
  - **No STUN/ICE hole punching.** NAT'd peers always go through the relay.
  - **No TLS or authentication** on tunnels or the relay. Any node can announce any address to a
    relay, so this is needed before untrusted deployments. This is M7's scope.
  - **Never run across a real NAT or WAN.** This belongs with M13.
  - **A NAT'd worker's `init` waits up to 30 s** for each peer tunnel, so the "two adjacent NAT'd
    workers" error takes 30 s to appear.
- **Carried (not done in M4; now scheduled in M9, M11 and M13):**
  - **Clients on several machines.** Multi-client sessions simulate every client in the data
    owner's process (one process holds all clients' data). That is enough for experiments, but
    there are no client nodes with their own data yet. Supporting them needs a worker-side data
    source, a per-batch return address on the shared server slice, and a client-state transfer
    RPC.
  - **Parallel mode.** The server consumes clients' batches one at a time. There are no per-client
    server copies and no batching across clients.
  - **Labels on another party.** Vertical SL (a label party that is not the data owner) is not
    supported. Every setting assumes the data owner holds inputs and labels.
  - **Sessions and fault tolerance.** Sessions refuse `fault_tolerance.enabled`, because recovery
    would not restore every client's state. A data-owner slice failure stops the run, by design.
  - **Attack metrics.** Only an observer hook (`Recorder`) and a doc example exist. A real attack
    (e.g. label inference from gradients, or model inversion) and a defence baseline belong to M11 (privacy suite) and M12 (baselines)
    (baselines: "one compression/attack paper").
  - **Earlier milestones, still open:**
    - compose/k8s never run, and multi-host bare metal is untested (M13);
    - the M0–M2 entry points and the hand-written root compose files are unchanged;
    - k8s discovery is always static;
    - the matrix is a script, not a CI integration test;
    - one matmul number for speed;
    - `uniform` ignores `valid_cuts()`;
    - recovery reloads the last *epoch* checkpoint.
- **Open owner decisions** (don't decide these alone):
  - license: GPL-3.0 or Apache-2.0;
  - whether to untrack the stale `lib/torchslicer/build/`;
  - when to merge into `main` and push (in order: M0, M1, M2, M3, M4);
  - Zenodo/GitHub integration;
  - rebuilding the Docker image (needed since M2; M4 changed the protos again).

## Known facts this is based on (verified in code, 2026-10-14)
- Centralized: the coordinator holds no layers and sends raw inputs to W0 → raw data leaves the data owner (fixed by M4 roles). (Since M4: still the default, `sl.setting: pipeline`; with `vanilla`/`u_shaped` the coordinator hosts the client slices.)
- P2P: the driver is worker 0 plus an embedded coordinator; it's an example script with no heartbeats/recovery. (Since M2: `ts.P2PExecutor`, with heartbeats and recovery.)
- Workers exit after every run: `DistributedExecutor.teardown()` sends `shutdown` to all of them, idle ones included.
- A centralized worker exits(1) if it can't register within ~30 attempts × 3 s.
- No U-shaped / multi-client / SplitFed in the library. (Since M4: `u_shaped`, `ts.ClientSession`; SplitFed's averaging is a user hook, `on_round_end`.) (Since M1, the built-in splitters are uniform, params, flops, profiled and named cuts.)
- SplitBud (Flower 1.9 fork, 2-tier, manual split, outbound-only clients) and SLPerf (MPI, manual split): see `sota.md`.
