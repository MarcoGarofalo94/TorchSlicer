# TorchSlicer roadmap (created 2026-10-14, updated 2026-10-15, end of M3)

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
- **M4, SL settings**: roles (the data owner hosts stage 0 so raw data stays local; label owner; U-shaped); `CutHook`
  (compression, quantization, noise, attack observers); built-in bytes-per-cut / client-compute / simulated-link metrics;
  multi-client sessions (sequential/parallel) with a client-weight handoff hook, no averaging in the core.
- **M5, NAT-friendly transport**: the gradient comes back in the reply (clients need only outbound connections); optional relay.
- **M6, baselines + paper**: `baselines/` template (config, seeds, expected numbers, CI smoke test, mean ± std over ≥3 seeds);
  seed it with vanilla SplitNN, U-shaped, SplitFed-via-Flower, and one compression/attack paper; framework paper vs SplitBud/SLPerf.

## Handoff (read this first if you are a new session)
Last updated 2026-10-15, at the end of M3 (version 0.5.0).
- **Where things stand:** M0 is done on `feat/m0-foundations`, M1 on `feat/m1-split-anything`, M2 on
  `feat/m2-p2p`, and M3 on `feat/m3-run-anywhere`. **Next: M4 (SL settings).** Create it with
  `git switch -c feat/m4-sl-settings feat/m3-run-anywhere`.
- **Read first:**
  - this file;
  - `notes/validation.md` (how correctness is checked, and every bug found so far, rounds 1–4);
  - `CLAUDE.md` (layout, commands, architecture, including P2P, planning and run-anywhere);
  - `docs/tutorials/p2p.md` and `docs/deploy/README.md`.
  - `notes/sota.md` and `notes/gaps.md` are for the *research* lines (`research/*` branches), not for framework milestones.
- **Every change must keep:**
  - `.venv/bin/python -m pytest tests/` green (142 tests at the end of M3, about 60 s);
  - the equivalence matrix green: `.venv/bin/python benchmarks/equivalence/run_matrix.py` (about 15 min on CPU;
    pass a model name for a quick check). Set `MATRIX_DEVICE=cuda` for the GPU, and
    `MATRIX_OPT='{"name":"AdamW","params":{"lr":0.001,"weight_decay":0.01}}'` for AdamW. A change to splitting
    must not change any loss. Every mode is compared at 1e-5 against full-precision losses.
- **Record as you go:** each bug found or behaviour changed goes into `notes/validation.md`, with the symptom, root
  cause, fix, cost and test, in the same commit as the fix. Note the pre-fix numbers *before* rerunning, because logs
  get overwritten. A paper on TorchSlicer will be written from these notes.
- **Environment:**
  - Git commits: the container has no git identity, so commit with `git -c user.name=... -c user.email=...`. Ask the
    owner for the values if your memory doesn't have them.
  - Downloads: HF hub and dataset downloads are blocked. Use random-init HF configs and synthetic data, as in
    `benchmarks/equivalence/tsmodels.py`.
  - No Docker and no kubectl in this container: compose/k8s output can be rendered and checked, not run.
    The host has a GPU, so `DEVICE`/`network.device: auto` means CUDA here. Pin `cpu` when comparing against a CPU reference.
  - gRPC stubs: regenerating them needs grpcio-tools 1.64.0 (so the gencode matches the declared minimum). It has no
    wheel for the venv's Python 3.13. What worked in M2: `pip download --only-binary=:all: --python-version 3.12
    --platform manylinux2014_x86_64 grpcio-tools==1.64.0 -d wheels`, unzip the wheels into a directory, and run
    `PYTHONPATH=<dir> /usr/bin/python3.12 -m grpc_tools.protoc ...` (the command in `CLAUDE.md`). Regenerating the
    unchanged protos this way gives byte-identical files. (M3 changed no proto.)
  - The `torchslicer` console script needs a reinstall of the editable package; `python -m torchslicer` works without it.
  - Don't commit the owner's untracked files at the repo root (`*.prompt.md`, `prompt*.txt`, `results/`,
    `sota_check.md`) or their `.gitignore` change. Launcher output goes to `runs/deploy/` (ignored).
- **Facts for M4 (verified in code, 2026-10-15):**
  - Naming: `ROLE` / `deploy.role` already means the *deployment* role (coordinator, worker, driver, follower,
    all). M4's SL roles (data owner, label owner) are a different axis. Give them their own name (e.g. `sl_role`
    or per-slice ownership in the plan) so they don't overload `ROLE`.
  - Data flow today: centralized: the coordinator sends **raw inputs to worker 0** and labels to the last worker
    (`DistributedExecutor._send_batch`). P2P: the driver keeps inputs and slice 0 in-process and sends labels to the
    last worker. Neither keeps labels on the data owner (no U-shaped). `first_slice` (declared capability) already
    pins a worker to slice 0, which is the hook for "the data owner hosts stage 0".
  - Every slice boundary carries exactly one tensor (`BaseSplitter.validate`). U-shaped needs the last slice (head
    + loss) back on the data owner, i.e. one node holding two non-adjacent slices; the executors assume node i holds slice i.
  - `ts.run` + `plan_nodes` are where a new role would be wired for deployment (one more `NodeSpec` role, env).
- **Carried (not done in M3):**
  - Compose and k8s output never ran (no Docker/kubectl here); multi-host bare metal not tested over a real network.
  - The M0–M2 entry points (`examples/train/centralized/GRPC/{coordinator,worker}`, `examples/train/hf`) and the
    hand-written root `docker-compose*.yml` (fixed 4 workers; P2P compose still passes the unused `WORKER_INDEX`)
    are unchanged. They could become `ts.run` scripts plus generated compose files. The P2P/LM scripts
    (`ts.run_p2p`) already work with the launcher, since `run_p2p` reads `ROLE`.
  - On k8s, discovery is always static (pod DNS order). A long-lived worker pool (StatefulSet + registration) is
    documented but not generated.
  - From M2: the matrix is still a script (not a CI integration test); the memory estimate ignores activations; one
    matmul number for speed; `uniform` ignores `valid_cuts()`; recovery reloads the last *epoch* checkpoint.
- **Open owner decisions** (don't decide these alone): license GPL-3.0 vs Apache-2.0; untracking the stale
  `lib/torchslicer/build/`; when to merge into `main` and push (M0 first, then M1, M2, M3); Zenodo/GitHub
  integration; rebuilding the Docker image (needed since M2: new protos and RPCs; generated compose/k8s use
  `deploy.image`).

## Known facts this is based on (verified in code, 2026-10-14)
- Centralized: the coordinator holds no layers and sends raw inputs to W0 → raw data leaves the data owner (fixed by M4 roles).
- P2P: the driver is worker 0 plus an embedded coordinator; it's an example script with no heartbeats/recovery. (Since M2: `ts.P2PExecutor`, with heartbeats and recovery.)
- Workers exit after every run: `DistributedExecutor.teardown()` sends `shutdown` to all of them, idle ones included.
- A centralized worker exits(1) if it can't register within ~30 attempts × 3 s.
- No U-shaped / multi-client / SplitFed in the library. (Since M1, the built-in splitters are uniform, params, flops, profiled and named cuts.)
- SplitBud (Flower 1.9 fork, 2-tier, manual split, outbound-only clients) and SLPerf (MPI, manual split): see `sota.md`.
