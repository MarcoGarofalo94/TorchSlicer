# TorchSlicer roadmap (created 2026-10-14, updated 2026-10-15, M6 done)

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
  running them is deferred to M8.
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
- **M7, baselines + paper**: `baselines/` template (config, seeds, expected numbers, CI smoke test, mean ± std over ≥3 seeds);
  seed it with vanilla SplitNN, U-shaped, SplitFed-via-Flower, and one compression/attack paper; framework paper vs SplitBud/SLPerf.
- **M8, deployment validation on a real environment** (deferred from M3 on 2026-10-15, owner decision): this container
  has no Docker or kubectl, so it runs in a fresh clone on a machine that has them, whenever one is available (it does
  not have to wait for M7). Rebuild the image (`make build-cpu`/`build-gpu`), then for both topologies run
  `launch --target compose` and `--target k8s` (kind/minikube is enough) on `examples/run_anywhere/` and check that
  the losses equal the `processes` target's; also run the bare-metal scripts across at least two real hosts. Record
  every bug in `notes/validation.md` as usual.

## Handoff (read this first if you are a new session)
Last updated 2026-10-15, M6 done (version 0.8.0).
- **Where things stand:**
  - Done, one branch per milestone: M0 on `feat/m0-foundations`, M1 on `feat/m1-split-anything`,
    M2 on `feat/m2-p2p`, M3 on `feat/m3-run-anywhere`, M4 on `feat/m4-sl-settings`.
    All milestone branches are pushed to `origin` (2026-10-15); `main` is unchanged.
  - M5 is on `feat/m5-offload`: implemented, with benchmarks deferred (see its Status entry).
  - M6 is on `feat/m6-nat` (done, not pushed).
  - **Next: M7 (baselines + paper).** Create it with
    `git switch -c feat/m7-baselines feat/m6-nat`.
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
  - `.venv/bin/python -m pytest tests/` green (247 tests at the end of M6, about 3 min;
    `test_sl.py::test_distributed_hooks_match_local[hook0]` flaked once under load);
  - the equivalence matrix green: `.venv/bin/python benchmarks/equivalence/run_matrix.py` (about
    15 min on CPU; pass a model name for a quick check).
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
    relay, so this is needed before untrusted deployments.
  - **Never run across a real NAT or WAN.** This belongs with M8.
  - **A NAT'd worker's `init` waits up to 30 s** for each peer tunnel, so the "two adjacent NAT'd
    workers" error takes 30 s to appear.
- **Carried (not done in M4):**
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
    (e.g. label inference from gradients, or model inversion) and a defence baseline belong to M7
    (baselines: "one compression/attack paper").
  - **Earlier milestones, still open:**
    - compose/k8s never run, and multi-host bare metal is untested (M8);
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
