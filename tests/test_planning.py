"""Capability-aware planning (M2): workers report what they are, the executor measures
the network, and the split is planned after discovery for those workers: faster
devices get more work, memory caps hold, and training still matches unsplit."""
import itertools
import random

import pytest
import torch
import torch.nn as nn

import torchslicer as ts
from torchslicer.config import RunConfig
from torchslicer.discovery.base import NodeInfo
from torchslicer.strategies.cost import balanced_cuts

from test_p2p import _cfg, _free_pair, _loader, _reference, _StopAfter


# ── the DP ────────────────────────────────────────────────────────────────────

def _brute(costs, valid, n, speeds=None, mem=None, caps=None, cut_cost=None, link_cost=None):
    best = None
    for cuts in itertools.combinations(valid, n - 1):
        bounds = [-1, *cuts, len(costs) - 1]
        times, ok = [], True
        for p in range(n):
            lo, hi = bounds[p] + 1, bounds[p + 1] + 1
            if caps and caps[p] is not None and sum(mem[lo:hi]) > caps[p]:
                ok = False
                break
            t = sum(costs[lo:hi]) / (speeds[p] if speeds else 1)
            if cut_cost and p < n - 1:
                t += cut_cost[bounds[p + 1]] * link_cost[p]
            times.append(t)
        if ok and (best is None or max(times) < best):
            best = max(times)
    return best


def _value(costs, cuts, n, speeds=None, cut_cost=None, link_cost=None):
    bounds = [-1, *cuts, len(costs) - 1]
    return max(sum(costs[bounds[p] + 1:bounds[p + 1] + 1]) / (speeds[p] if speeds else 1)
               + (cut_cost[bounds[p + 1]] * link_cost[p] if cut_cost and p < n - 1 else 0)
               for p in range(n))


def test_heterogeneous_dp_matches_brute_force():
    rng = random.Random(0)
    for _ in range(300):
        L = rng.randint(3, 9)
        n = rng.randint(1, min(4, L))
        costs = [rng.uniform(0.1, 5) for _ in range(L)]
        valid = sorted(rng.sample(range(L - 1), rng.randint(n - 1, L - 1)))
        kw = {"speeds": [rng.uniform(0.5, 4) for _ in range(n)]}
        if rng.random() < 0.5:
            kw["mem"] = [rng.uniform(0, 3) for _ in range(L)]
            kw["caps"] = [rng.choice([None, rng.uniform(2, 10)]) for _ in range(n)]
        if rng.random() < 0.5:
            kw["cut_cost"] = [rng.uniform(0, 2) for _ in range(L)]
            kw["link_cost"] = [rng.uniform(0, 2) for _ in range(n - 1)]
        want = _brute(costs, valid, n, **kw)
        if want is None:
            with pytest.raises(ValueError, match="memory caps"):
                balanced_cuts(costs, valid, n, **kw)
            continue
        cuts = balanced_cuts(costs, valid, n, **kw)
        got = _value(costs, cuts, n, kw["speeds"], kw.get("cut_cost"), kw.get("link_cost"))
        assert got == pytest.approx(want)


def test_faster_device_gets_proportionally_more():
    # 8 equal nodes, the second device is 3x faster: 2 + 6 balances at 2 time units each
    assert balanced_cuts([1.0] * 8, list(range(7)), 2, speeds=[1, 3]) == [1]
    assert balanced_cuts([1.0] * 8, list(range(7)), 2) == [3]


# ── strategies with nodes ───────────────────────────────────────────────────────

def _mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(),
                         nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))


def _nodes(*gflops, caps=None):
    caps = caps or [0] * len(gflops)
    return [NodeInfo(node_id=str(i), address=str(i), compute_gflops=g, max_memory_mb=c)
            for i, (g, c) in enumerate(zip(gflops, caps))]


def test_identical_workers_with_noisy_speeds_keep_the_homogeneous_plan():
    graph = ts.trace(_mlp())
    s = ts.ParamSplitter()
    homogeneous = [p.layer_indices for p in s.split(graph, 3)]
    noisy = [p.layer_indices for p in s.split(graph, 3, nodes=_nodes(100, 112, 95))]
    assert noisy == homogeneous and "speeds" not in s.last_terms


def _wide():
    """Two 512x512 layers of ~4 MB training memory each (weights, grads, Adam states)."""
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                         nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 4))


def test_params_plan_follows_speed():
    graph = ts.trace(_mlp())
    fast_last = ts.ParamSplitter().split(graph, 2, nodes=_nodes(10, 40))
    fast_first = ts.ParamSplitter().split(graph, 2, nodes=_nodes(40, 10))
    assert len(fast_last[0].layer_indices) < len(fast_first[0].layer_indices)


def test_params_plan_respects_memory_caps():
    graph = ts.trace(_wide())
    s = ts.ParamSplitter()
    mb = [m / 2**20 for m in s.node_memory(graph)]
    assert 4.0 < mb[2] < 4.1 and 4.0 < mb[4] < 4.1
    # node 0 is 4x faster: unconstrained, it takes both wide layers
    free = s.split(graph, 2, nodes=_nodes(40, 10))
    assert {2, 4} <= set(free[0].layer_indices)
    # capped at 5 MB, it can hold only one of them
    capped = s.split(graph, 2, nodes=_nodes(40, 10, caps=[5, 0]))
    assert sum(mb[j] for j in capped[0].layer_indices) <= 5
    assert 2 in capped[0].layer_indices and 4 not in capped[0].layer_indices
    with pytest.raises(ValueError, match="memory caps"):
        s.split(graph, 2, nodes=_nodes(10, 10, caps=[1, 1]))


def test_flops_plan_adds_link_time_when_links_are_measured():
    x = torch.randn(4, 8)
    graph = ts.trace(_mlp())
    s = ts.FlopsSplitter(example_input=x)
    nodes = _nodes(50, 50, 50)
    for nd in nodes[:-1]:
        nd.link_bandwidth_mbps = 100.0
    s.split(graph, 3, nodes=nodes)
    assert {"speeds", "cut_cost", "link_cost"} <= set(s.last_terms)
    # without link measurements: identical speeds → homogeneous plan
    s2 = ts.FlopsSplitter(example_input=x)
    plain = s2.split(graph, 3)
    assert [p.layer_indices for p in s2.split(graph, 3, nodes=_nodes(50, 50, 50))] == \
        [p.layer_indices for p in plain]


# ── end to end: P2P on simulated heterogeneous nodes ────────────────────────────

def _node_cfg(base, gflops, max_memory_mb=0):
    cfg = RunConfig()
    for section in ("training", "transport", "fault_tolerance", "logging", "checkpoint", "planning"):
        setattr(cfg, section, getattr(base, section))
    cfg.run_id = base.run_id
    cfg.network.worker_compute_gflops = gflops
    cfg.network.worker_max_memory_mb = max_memory_mb
    return cfg


def _reference_for(model_fn, epochs):
    model = model_fn()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    out = []
    for _ in range(epochs):
        losses = []
        for x, y in _loader():
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        out.append(sum(losses) / len(losses))
    return out


def _hetero_run(base, speeds, caps=(0, 0, 0), strategy="params", callbacks=None, model=_mlp):
    followers = [ts.start_worker(_free_pair(), host="127.0.0.1",
                                 run_config=_node_cfg(base, g, c))
                 for g, c in zip(speeds[1:], caps[1:])]
    addr = f"127.0.0.1:{_free_pair()}"
    driver_cfg = _node_cfg(base, speeds[0], caps[0])
    ex = ts.P2PExecutor(peers=[w.address for w in followers], address=addr, bind_addr=addr,
                        run_config=driver_cfg)
    sliced = ts.slice(model(), n=len(speeds), strategy=strategy, executor=ex)
    try:
        history = sliced.train(_loader(), run_config=driver_cfg, callbacks=callbacks)
    finally:
        for w in followers:
            w.stop()
    return sliced, ex, history, followers


def test_p2p_plan_follows_worker_speeds_and_training_matches_unsplit():
    base = _cfg(epochs=2)
    eager = ts.slice(_mlp(), n=3, strategy="params").partitions
    sliced, ex, history, _ = _hetero_run(base, speeds=[10.0, 10.0, 80.0])
    assert [nd.compute_gflops for nd in ex._nodes] == [10.0, 10.0, 80.0]
    sizes = [len(p.layer_indices) for p in sliced.partitions]
    assert sizes[2] > len(eager[2].layer_indices)          # the fast node got more
    assert [p.layer_indices for p in sliced.partitions] != [p.layer_indices for p in eager]
    assert [h["loss"] for h in history] == pytest.approx(_reference(2), abs=1e-5)


def test_p2p_plan_respects_a_worker_memory_cap():
    base = _cfg(epochs=1)
    # follower 1 is 8x faster but declares a 5 MB cap: it gets one wide layer, not both
    sliced, ex, history, _ = _hetero_run(base, speeds=[10.0, 80.0, 10.0], caps=(0, 5, 0),
                                         model=_wide)
    assert ex._nodes[1].max_memory_mb == 5
    mb = [m / 2**20 for m in ts.ParamSplitter().node_memory(ts.trace(_wide()))]
    assert sum(mb[j] for j in sliced.partitions[1].layer_indices) <= 5
    free, *_ = _hetero_run(base, speeds=[10.0, 80.0, 10.0], model=_wide)
    assert sum(mb[j] for j in free.partitions[1].layer_indices) > 5
    assert [h["loss"] for h in history] == pytest.approx(_reference_for(_wide, 1), abs=1e-5)


def test_recovery_replans_with_the_run_strategy_for_the_survivors(tmp_path):
    base = _cfg(epochs=3, enabled=True, max_retries=1)
    base.checkpoint.dir = str(tmp_path)
    holder = {}

    class StopLast(_StopAfter):
        def __init__(self):
            pass

        def on_epoch_end(self, epoch, metrics):
            if epoch == 1:
                holder["followers"][1].stop()
            return metrics

    # start followers first so the callback can stop one
    followers = [ts.start_worker(_free_pair(), host="127.0.0.1", run_config=_node_cfg(base, g))
                 for g in (80.0, 10.0)]
    holder["followers"] = followers
    addr = f"127.0.0.1:{_free_pair()}"
    cfg = _node_cfg(base, 10.0)
    ex = ts.P2PExecutor(peers=[w.address for w in followers], address=addr, bind_addr=addr,
                        run_config=cfg)
    sliced = ts.slice(_mlp(), n=3, strategy="params", executor=ex)
    try:
        sliced.train(_loader(), run_config=cfg, callbacks=[StopLast()])
    finally:
        for w in followers:
            w.stop()
    survivors = _nodes(10.0, 80.0)
    want = ts.ParamSplitter().split(ts.trace(_mlp()), 2, nodes=survivors)
    assert [p.layer_indices for p in sliced.partitions] == [p.layer_indices for p in want]


def test_links_are_measured_between_neighbours():
    base = _cfg(epochs=1)
    base.planning.measure_links = True
    _, ex, history, _ = _hetero_run(base, speeds=[10.0, 10.0, 10.0])
    assert all(nd.link_bandwidth_mbps > 0 and nd.link_rtt_ms > 0 for nd in ex._nodes[:-1])
    assert ex._nodes[-1].link_bandwidth_mbps == 0
    assert all(nd.bandwidth_mbps > 0 for nd in ex._nodes[1:])   # driver → followers


# ── centralized: registration carries capabilities, first_slice goes first ─────

def test_registration_carries_capabilities():
    from torchslicer.executors.distributed import _CoordinatorServicer, DistributedExecutor
    from torchslicer.transport.grpc.coordinator import coordinator_service_pb2 as pb
    disc = ts.CoordinatorDiscovery(run_id="r")
    ex = DistributedExecutor(discovery=disc, coordinator_addr="127.0.0.1:1")
    _CoordinatorServicer(ex).register(pb.RegisterRequest(node=pb.NodeInfo(
        node_id="w", address="w:1", compute_gflops=42.5, max_memory_mb=512, first_slice=True)), None)
    nd = disc.discover(1, timeout=1)[0]
    assert (nd.compute_gflops, nd.max_memory_mb, nd.first_slice) == (42.5, 512, True)


def test_first_slice_worker_is_ordered_first():
    base = _cfg(epochs=1)
    cfgs = [_node_cfg(base, 0.0) for _ in range(3)]
    cfgs[2].network.worker_first_slice = True
    workers = [ts.start_worker(_free_pair(), host="127.0.0.1", run_config=c) for c in cfgs]
    coord = f"127.0.0.1:{_free_pair()}"
    try:
        ex = ts.DistributedExecutor(discovery=ts.StaticDiscovery([w.address for w in workers]),
                                    coordinator_addr=coord, coordinator_bind_addr=coord,
                                    run_config=base)
        history = ts.slice(_mlp(), n=3, executor=ex).train(_loader(), run_config=base)
    finally:
        for w in workers:
            w.stop()
    assert ex._nodes[0].address == workers[2].address
    assert all(nd.compute_gflops > 0 for nd in ex._nodes)     # measured
    assert [h["loss"] for h in history] == pytest.approx(_reference(1), abs=1e-5)
