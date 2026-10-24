"""Clients as nodes (``sl.client_nodes``, M10): each client is a worker holding slice 0 and
its own data; the coordinator holds neither.

With one batch in flight, a run on client nodes must train exactly like the in-process
``ClientSession`` on the same clients: same losses, same final client weights, for every
schedule (sequential with each handoff, parallel, an aggregating ``Algorithm``, client
sampling), every transport and GPipe micro-batches; also through ``ts.run`` and as real OS
processes.
"""
import json
import pathlib
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

import torchslicer as ts
from torchslicer.config import RunConfig
from torchslicer.deploy.launch import render_k8s
from torchslicer.deploy.nodes import node_names, plan_nodes
from torchslicer.deploy.runner import free_port
from torchslicer.discovery import StaticDiscovery

ROOT = pathlib.Path(__file__).resolve().parent.parent
C, CUTS = 3, ["1", "3"]      # 3 clients; 3 slices: Linear+ReLU | Linear+ReLU | Linear


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))


def _loaders():
    out = []
    for k in range(C):
        g = torch.Generator().manual_seed(100 + k)
        out.append([(torch.randn(8, 8, generator=g), torch.randint(0, 4, (8,), generator=g))
                    for _ in range(3 + k)])          # unequal clients
    return out


def _cfg(transport="grpc"):
    c = RunConfig()
    c.training.epochs = 2
    c.training.optimizer = {"name": "SGD", "params": {"lr": 0.05, "momentum": 0.9}}
    c.training.criterion = {"name": "CrossEntropyLoss", "params": {}}
    c.logging.enabled = False
    c.network.device = "cpu"
    c.transport.tensor = transport
    c.sl.setting = "vanilla"
    c.sl.clients = C
    c.discovery.timeout = 30
    return c


def _local(mode, handoff, algorithm, n_micro=1, loaders=None, eval_data=None, history=False):
    session = ts.ClientSession(loaders or _loaders(), mode=mode, handoff=handoff)
    h = ts.slice(_model(), cuts=CUTS).train(session, run_config=_cfg(), algorithm=algorithm,
                                           n_micro_batches=n_micro, eval_data=eval_data)
    return ([e["loss"] for e in h], session.states) + ((h,) if history else ())


def _shuffled():
    """The clients' data as shuffling DataLoaders, each with its own generator."""
    from torch.utils.data import DataLoader, TensorDataset
    return [DataLoader(TensorDataset(torch.cat([x for x, _ in b]), torch.cat([y for _, y in b])), batch_size=8,
                       shuffle=True, generator=torch.Generator().manual_seed(k))
            for k, b in enumerate(_loaders())]


def _nodes(mode, handoff, algorithm, transport="grpc", n_micro=1, servicers=None, server_servicer=None,
           eval_data=None, registered=False, loaders=None, **sl):
    cfg = _cfg(transport)
    for key, v in sl.items():
        setattr(cfg.sl, key, v)
    cfg.fault_tolerance.heartbeat_interval_s = 0.5
    offsets = cfg.transport.listen_offsets()
    servers = [ts.start_worker(free_port(offsets), host="127.0.0.1", run_config=cfg,
                               servicer_class=server_servicer) for _ in range(2)]
    data = loaders or _loaders()
    clients = [ts.start_worker(free_port(offsets), host="127.0.0.1", run_config=cfg, data=data[k],
                               client_index=k, servicer_class=(servicers or {}).get(k))
               for k in range(C)]
    peers = [w.address for w in clients[::-1] + servers]      # discovery order doesn't matter
    addr = f"127.0.0.1:{free_port(cfg.transport.listen_offsets(drives=True))}"
    discovery = (_Registered(peers, {c.address: ["client", f"client={k}"] for k, c in enumerate(clients)})
                 if registered else StaticDiscovery(peers=peers))
    ex = ts.ClientNodesExecutor(discovery=discovery, coordinator_addr=addr, run_config=cfg)
    session = ts.RemoteSession(C, mode=mode, handoff=handoff)
    try:
        h = ts.slice(_model(), cuts=CUTS, executor=ex).train(session, run_config=cfg, algorithm=algorithm,
                                                            n_micro_batches=n_micro, eval_data=eval_data)
        states = session.states          # every client's final state, read at teardown
    finally:
        for w in servers + clients:
            w.stop()
    return [e["loss"] for e in h], states, h


class _Registered(StaticDiscovery):
    """The nodes as coordinator discovery keeps them: with the tags they registered with,
    also after they go down."""
    def __init__(self, peers, tags):
        super().__init__(peers)
        for nd in self._nodes:
            nd.tags = list(tags.get(nd.address, []))


def _swap_first_two(prev_state, next_state, info):   # a custom handoff
    return next_state if next_state is not None else prev_state


class _Odd(ts.Algorithm):                              # client sampling: odd rounds skip client 1
    def select_clients(self, available, round):
        return [k for k in available if not (round % 2 and k == 1)]


class _OddV1(ts.SplitFedV1Algorithm):                  # SplitFed V1 with client sampling
    def select_clients(self, available, round):
        return [k for k in available if not (round % 2 and k == 1)]


def _sl_example():
    """examples/sl/train.py, under another name: it imports run_anywhere's ``train``."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sl_example_train", ROOT / "examples/sl/train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sl_example_losses(cfg):
    ex = _sl_example()
    return [e["loss"] for e in ts.run(ex.build_model, lambda: ex.get_data(cfg), run_config=cfg,
                                      cuts=["pool1", "pool2"], verbose=False)]


def _local_twin():
    """experiments/sl_client_nodes.yaml simulated in one process (a ClientSession)."""
    cfg = ts.RunConfig.load(str(ROOT / "experiments/sl_client_nodes.yaml"))
    cfg.deploy.topology, cfg.sl.client_nodes = "local", False
    return cfg


def _same_states(a, b):
    for sa, sb in zip(a, b):
        for k, v in sa[0]["model"].items():
            assert torch.allclose(v, sb[0]["model"][k], atol=1e-6), k


@pytest.mark.parametrize("mode,handoff,algorithm,transport", [
    ("sequential", "copy", None, "grpc"),
    ("sequential", "keep", None, "tcp"),
    ("sequential", _swap_first_two, None, "grpc"),
    ("parallel", "copy", None, "tcp"),
    ("parallel", "copy", ts.SplitFedAlgorithm(), "grpc"),
    ("parallel", "copy", _Odd(), "grpc"),
    ("parallel", "copy", ts.SplitFedV1Algorithm(), "tcp"),       # per-client server copies
    ("sequential", "keep", ts.SplitFedV1Algorithm(), "grpc"),
    ("parallel", "copy", _OddV1(), "grpc"),
])
def test_client_nodes_match_the_in_process_session(mode, handoff, algorithm, transport):
    ref, ref_states = _local(mode, handoff, algorithm)
    got, states, h = _nodes(mode, handoff, algorithm, transport)
    assert [round(x, 6) for x in got] == ref
    _same_states(states, ref_states)
    moved = h[-1]["bytes_state"]
    aggregates = isinstance(algorithm, (ts.SplitFedAlgorithm, ts.SplitFedV1Algorithm))
    if (handoff == "keep" or mode == "parallel") and not aggregates:
        assert moved == 0              # nothing needs a client's state elsewhere
    else:
        assert moved > 0               # a handoff or an aggregation moved states
    assert h[-1]["bytes_label"] > 0    # labels go from the clients to the last slice


@pytest.mark.parametrize("mode", ["parallel", "sequential"])
def test_client_nodes_with_shuffled_data_match_the_in_process_session(mode):
    # a client node took exactly its n batches with next(); a for loop (the in-process
    # session) also runs the iterator to its end, where RandomSampler draws its generator
    # once more, so from epoch 2 on the nodes trained on another order
    ref = _local(mode, "copy", None, loaders=_shuffled())[0]
    assert [round(x, 6) for x in _nodes(mode, "copy", None, loaders=_shuffled())[0]] == ref


@pytest.mark.parametrize("algorithm", [ts.SplitFedAlgorithm, ts.SplitFedV1Algorithm])
def test_client_nodes_with_gpipe_micro_batches(algorithm):
    ref, ref_states = _local("parallel", "copy", algorithm(), n_micro=2)
    got, states, _ = _nodes("parallel", "copy", algorithm(), "tcp", n_micro=2)
    assert [round(x, 6) for x in got] == ref
    _same_states(states, ref_states)


def test_slice_one_returns_each_gradient_to_its_client():
    w = ts.WorkerServicer()
    w._n_micro, w._prev_peers = 2, [("a", "stub-a", None), ("b", "stub-b", None), ("c", "stub-c", None)]
    # batch b of client k has id b * 3 + k; micro-batch m of it has id (b * 3 + k) * 2 + m
    assert [w._prev_for(((b * 3 + k) * 2 + m))[0] for b in range(2) for k in range(3)
            for m in range(2)] == list("aabbcc" * 2)


def test_ts_run_with_client_nodes_matches_the_local_session():
    cfg = ts.RunConfig.load(str(ROOT / "experiments/sl_client_nodes.yaml"))
    assert [round(x, 6) for x in _sl_example_losses(cfg)] == _sl_example_losses(_local_twin())


def test_launch_processes_runs_client_nodes(tmp_path):
    r = subprocess.run([sys.executable, "-m", "torchslicer", "launch", "examples/sl/train.py", "-c",
                        "experiments/sl_client_nodes.yaml", "--target", "processes", "-o", str(tmp_path)],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert sum(l.startswith("[launch] client-") for l in r.stdout.splitlines()) == 4   # a process each
    got = json.loads(next(l for l in r.stdout.splitlines() if l.startswith("HISTORY "))[8:])
    assert [round(x, 6) for x in got] == _sl_example_losses(_local_twin())


def test_node_plan_has_one_node_per_client():
    cfg = ts.RunConfig.load(str(ROOT / "experiments/sl_client_nodes.yaml"))
    names = node_names(cfg)
    assert [r for _, r in names] == ["coordinator", "worker", "worker"] + ["client"] * 4
    nodes = plan_nodes(cfg, ["h"] * len(names), list(range(30000, 30000 + len(names))),
                       discovery="static")
    assert [nd.env.get("CLIENT_INDEX") for nd in nodes[3:]] == ["0", "1", "2", "3"]
    assert len(nodes[0].env["WORKER_PEERS"].split(",")) == 6     # the coordinator dials all
    with pytest.raises(ValueError, match="k8s"):
        render_k8s(cfg, "train.py", "")


def test_client_nodes_refuse_what_they_cant_run():
    cfg = _cfg()
    cfg.transport.tensor = "nccl"
    with pytest.raises(ValueError, match="nccl"):
        ts.ClientNodesExecutor(discovery=StaticDiscovery(peers=[]), coordinator_addr="127.0.0.1:1",
                               run_config=cfg)
    cfg = _cfg()
    cfg.sl.setting = "u_shaped"
    with pytest.raises(ValueError, match="vanilla"):
        ts.ClientNodesExecutor(discovery=StaticDiscovery(peers=[]), coordinator_addr="127.0.0.1:1",
                               run_config=cfg)
    cfg = _cfg()
    cfg.deploy.role, cfg.deploy.topology = "client", "centralized"
    with pytest.raises(ValueError, match="client_nodes"):
        ts.run(_model, _loaders, run_config=cfg)
    with pytest.raises(RuntimeError, match="no client nodes"):
        len(ts.RemoteSession(2))


def test_too_few_client_nodes_fail_with_the_counts():
    cfg = _cfg()                                   # 3 clients + 2 server slices = 5 nodes
    offsets = cfg.transport.listen_offsets()
    workers = [ts.start_worker(free_port(offsets), host="127.0.0.1", run_config=cfg) for _ in range(4)]
    client = ts.start_worker(free_port(offsets), host="127.0.0.1", run_config=cfg, data=_loaders()[0],
                             client_index=0)
    addr = f"127.0.0.1:{free_port(cfg.transport.listen_offsets(drives=True))}"
    ex = ts.ClientNodesExecutor(discovery=StaticDiscovery(peers=[client.address, *(w.address for w in workers)]),
                                coordinator_addr=addr, run_config=cfg)
    try:
        with pytest.raises(ValueError, match="needs 3 client nodes and 2 server workers; got 1 and 4"):
            ts.slice(_model(), cuts=CUTS, executor=ex).train(ts.RemoteSession(C), run_config=cfg)
    finally:
        for w in workers + [client]:
            w.stop()


# ── step 4: client nodes that die, rounds with a deadline ───────────────────────

def _dies_at(epoch, index):
    """A client node that goes down when asked for batch ``index`` of ``epoch``."""
    class _Dies(ts.WorkerServicer):
        def client_step(self, request, context):
            if (request.epoch, request.index) == (epoch, index):
                self._server.stop(0)
                raise RuntimeError("client node killed")
            return super().client_step(request, context)
    return _Dies


class _Without(ts.Algorithm):          # the in-process twin of a client that left in round 1
    def __init__(self, k):
        self.k = k

    def select_clients(self, available, round):
        return [c for c in available if round == 1 or c != self.k]


@pytest.mark.parametrize("mode,dies_at", [("parallel", 1), ("sequential", 0)])
def test_a_dead_client_leaves_and_the_others_go_on(mode, dies_at):
    # client 1 dies when asked for its batch `dies_at` of round 1: it trained `dies_at`
    # batches, then takes no part, so the twin is a session where it had only those
    loaders = _loaders()
    loaders[1] = loaders[1][:dies_at]
    ref, ref_states = _local(mode, "copy", _Without(1), loaders=loaders)
    got, states, h = _nodes(mode, "copy", None, servicers={1: _dies_at(1, dies_at)})
    assert [round(x, 6) for x in got] == ref
    _same_states([states[0], states[2]], [ref_states[0], ref_states[2]])
    assert h[0]["clients_lost"] == [1] and h[1]["clients_lost"] == [1]
    n1 = len(_loaders()[1])
    assert h[0]["batches_skipped"] == n1 - dies_at - 1 and h[1]["batches_skipped"] == n1


def test_a_dead_registered_client_is_not_asked_to_join_again():
    # coordinator discovery keeps a dead client's registered tags: the next round start
    # took it for a returning client, its join timed out with TimeoutError and the run died
    got, _, h = _nodes("parallel", "copy", None, servicers={1: _dies_at(1, 1)}, registered=True)
    loaders = _loaders()
    loaders[1] = loaders[1][:1]
    assert [round(x, 6) for x in got] == _local("parallel", "copy", _Without(1), loaders=loaders)[0]
    assert h[1]["clients_lost"] == [1]


def test_a_dead_client_stops_the_run_without_drop_failed_clients():
    with pytest.raises(Exception):
        _nodes("parallel", "copy", None, servicers={1: _dies_at(1, 1)}, drop_failed_clients=False)


def test_a_dead_client_is_left_out_of_the_aggregation():
    got, states, h = _nodes("parallel", "copy", ts.SplitFedAlgorithm(), servicers={2: _dies_at(1, 2)})
    assert h[-1]["clients_lost"] == [2] and len(got) == 2
    _same_states([states[0]], [states[1]])          # the survivors hold the same aggregate


def test_a_round_deadline_skips_the_rest_of_the_round():
    class _Slow(ts.WorkerServicer):
        def client_step(self, request, context):
            import time
            time.sleep(0.2)
            return super().client_step(request, context)
    got, _, h = _nodes("sequential", "copy", None, servicers={k: _Slow for k in range(C)},
                       round_deadline_s=0.5)
    total = sum(len(l) for l in _loaders())
    assert all(0 < e["batches_skipped"] < total for e in h)   # it ran some, not all


# ── step 4b: clients that join late or come back ─────────────────────────────────

class _PerRound:
    """A client's data that differs per round (an in-process twin of a node restarted with
    its full loader): round r iterates ``rounds[min(r, last)]``."""

    def __init__(self, rounds):
        self.rounds, self.r = rounds, -1

    def __iter__(self):
        self.r += 1
        return iter(self.rounds[min(self.r, len(self.rounds) - 1)])

    def __len__(self):
        return len(self.rounds[-1])


class _Join(ts.Algorithm):
    """Twin: client ``k`` is absent in round 1; with ``reset``, it restarts round 2 from the
    session's initial state (``reset`` = the session); ``base`` adds an aggregation."""

    def __init__(self, k, reset=None, base=None):
        self.k, self.reset, self.base = k, reset, base

    def select_clients(self, available, round):
        return [c for c in available if round != 1 or c != self.k or self.reset is not None]

    def on_round_end(self, states, metrics, round):
        out = self.base.on_round_end(states, metrics, round) if self.base else None
        if self.reset is not None and round == 1:
            out = list(out or states)
            out[self.k] = self.reset._initial
        return out

    def aggregate(self, client_states, server_states, round):
        return self.base.aggregate(client_states, server_states, round) if self.base else (None, None)


def _late_run(algorithm, k, servicer=None, **sl):
    """Client nodes where client ``k`` starts (again) at the end of round 1."""
    cfg = _cfg()
    for key, v in sl.items():
        setattr(cfg.sl, key, v)
    cfg.fault_tolerance.heartbeat_interval_s = 0.5
    offsets = cfg.transport.listen_offsets()
    servers = [ts.start_worker(free_port(offsets), host="127.0.0.1", run_config=cfg) for _ in range(2)]
    ports = [free_port(offsets) for _ in range(C)]
    start = lambda c, cls=None: ts.start_worker(ports[c], host="127.0.0.1", run_config=cfg,
                                                data=_loaders()[c], client_index=c, servicer_class=cls)
    workers = [start(c, servicer if c == k else None) if servicer or c != k else None for c in range(C)]

    class _Restart(ts.TrainingCallback):
        def on_epoch_end(self, epoch, metrics):
            if epoch == 1:
                if workers[k] is not None:
                    workers[k].stop()
                workers[k] = start(k)
            return metrics

    peers = [w.address for w in servers] + [f"127.0.0.1:{p}" for p in ports]
    addr = f"127.0.0.1:{free_port(cfg.transport.listen_offsets(drives=True))}"
    ex = ts.ClientNodesExecutor(discovery=StaticDiscovery(peers=peers), coordinator_addr=addr, run_config=cfg)
    session = ts.RemoteSession(C, mode="parallel")
    try:
        h = ts.slice(_model(), cuts=CUTS, executor=ex).train(session, run_config=cfg, algorithm=algorithm,
                                                            callbacks=[_Restart()])
    finally:
        for w in servers + workers:
            if w is not None:
                w.stop()
    return [e["loss"] for e in h], session.states, h


@pytest.mark.parametrize("aggregate", [False, True])
def test_a_late_client_joins_at_the_next_round(aggregate):
    base = ts.SplitFedAlgorithm if aggregate else None
    ref, ref_states = _local("parallel", "copy", _Join(2, base=base() if base else None))
    got, states, h = _nodes_late = _late_run(base() if base else None, 2, min_clients=2)
    assert [round(x, 6) for x in got] == ref
    _same_states(states, ref_states)
    assert h[0]["clients_lost"] == [2] and h[1]["clients_joined"] == [2] and "clients_lost" not in h[1]


def test_a_client_that_died_comes_back_from_the_initial_weights():
    loaders = _loaders()
    loaders[1] = _PerRound([loaders[1][:1], loaders[1]])   # 1 batch in round 1, all in round 2
    session = ts.ClientSession(loaders, mode="parallel")
    ref = [e["loss"] for e in ts.slice(_model(), cuts=CUTS).train(
        session, run_config=_cfg(), algorithm=_Join(1, reset=session))]
    got, states, h = _late_run(None, 1, servicer=_dies_at(1, 1))
    assert [round(x, 6) for x in got] == ref
    _same_states(states, session.states)
    assert h[0]["clients_lost"] == [1] and h[1]["clients_joined"] == [1]


# ── step 4c: a server worker dies (fault_tolerance.enabled) ─────────────────────

@pytest.mark.parametrize("mode,kill_at,opt", [("parallel", 2, "SGD"), ("sequential", 2, "SGD"),
                                              ("parallel", 1, "SGD"), ("parallel", 2, "AdamW")])
def test_server_recovery_replays_the_round_exactly(mode, kill_at, opt, tmp_path):
    # kill_at 2: back to round 1's checkpoint; kill_at 1: no checkpoint yet, back to the start.
    # The server slices are re-split with their optimizer state (momentum, Adam moments and
    # step), so the losses and client weights are the uninterrupted run's
    def cfg_():
        c = _cfg()
        c.training.epochs = 3
        c.training.optimizer = {"name": opt, "params": {"lr": 0.05, **({"momentum": 0.9} if opt == "SGD" else {})}}
        return c
    session = ts.ClientSession(_loaders(), mode=mode)
    ref = [e["loss"] for e in ts.slice(_model(), cuts=CUTS).train(session, run_config=cfg_())]

    cfg = cfg_()
    cfg.fault_tolerance.enabled, cfg.fault_tolerance.max_retries = True, 1
    cfg.fault_tolerance.heartbeat_interval_s = 0.5
    cfg.checkpoint.dir = str(tmp_path)
    offsets = cfg.transport.listen_offsets()
    servers = [ts.start_worker(free_port(offsets), host="127.0.0.1", run_config=cfg) for _ in range(2)]
    clients = [ts.start_worker(free_port(offsets), host="127.0.0.1", run_config=cfg, data=_loaders()[k],
                               client_index=k) for k in range(C)]

    class _KillServer(ts.TrainingCallback):          # the last server slice dies as round 2 starts
        def on_epoch_begin(self, epoch):
            if epoch == kill_at and servers[1] is not None:
                servers[1].stop()

    addr = f"127.0.0.1:{free_port(cfg.transport.listen_offsets(drives=True))}"
    ex = ts.ClientNodesExecutor(discovery=StaticDiscovery(peers=[w.address for w in clients + servers]),
                                coordinator_addr=addr, run_config=cfg)
    remote = ts.RemoteSession(C, mode=mode)
    try:
        h = ts.slice(_model(), cuts=CUTS, executor=ex).train(remote, run_config=cfg, callbacks=[_KillServer()])
    finally:
        for w in servers + clients:
            w.stop()
    assert [round(e["loss"], 6) for e in h] == ref
    _same_states(remote.states, session.states)
    assert len(ex.partitions) == 2                    # slice 0 on the clients, one server worker left


def test_server_recovery_mid_round_with_shuffled_data_replays_exactly(tmp_path):
    # a server worker dies in round 2 after some batches: the clients' retried epoch must
    # start from the same random state (shuffle), also though recovery re-inits them, and
    # the server slices come back with their optimizer state
    def cfg_():
        c = _cfg()
        c.training.epochs = 3
        return c                                          # SGD with momentum
    session = ts.ClientSession(_shuffled(), mode="parallel")
    ref = [e["loss"] for e in ts.slice(_model(), cuts=CUTS).train(session, run_config=cfg_(),
                                                                  algorithm=ts.SplitFedAlgorithm())]
    cfg = cfg_()
    cfg.fault_tolerance.enabled, cfg.fault_tolerance.max_retries = True, 1
    cfg.fault_tolerance.heartbeat_interval_s = 0.5
    cfg.checkpoint.dir = str(tmp_path)
    offsets = cfg.transport.listen_offsets()
    servers = [ts.start_worker(free_port(offsets), host="127.0.0.1", run_config=cfg) for _ in range(2)]

    class _KillsServer(ts.WorkerServicer):               # client 1's batch 1 of round 2
        def client_step(self, request, context):
            if (request.epoch, request.index) == (2, 1) and servers[1] is not None:
                servers[1].stop()
            return super().client_step(request, context)
    data = _shuffled()
    clients = [ts.start_worker(free_port(offsets), host="127.0.0.1", run_config=cfg, data=data[k], client_index=k,
                               servicer_class=_KillsServer if k == 1 else None) for k in range(C)]
    addr = f"127.0.0.1:{free_port(cfg.transport.listen_offsets(drives=True))}"
    ex = ts.ClientNodesExecutor(discovery=StaticDiscovery(peers=[w.address for w in clients + servers]),
                                coordinator_addr=addr, run_config=cfg)
    remote = ts.RemoteSession(C, mode="parallel")
    try:
        h = ts.slice(_model(), cuts=CUTS, executor=ex).train(remote, run_config=cfg, algorithm=ts.SplitFedAlgorithm())
    finally:
        for w in servers + clients:
            w.stop()
    assert [round(e["loss"], 6) for e in h] == ref
    assert len(ex.partitions) == 2


def test_profile_breakdown_sums_every_client(monkeypatch):
    # profile.verbosity >= 1: each epoch says where its time went, per slice and phase; slice
    # 0's phases are summed over the client nodes (each client runs its own batches)
    plain = _cfg
    def profiled(transport="grpc"):
        c = plain(transport)
        c.profile.verbosity = 1
        return c
    monkeypatch.setattr(sys.modules[__name__], "_cfg", profiled)
    _, _, h = _nodes("parallel", "copy", None)
    for e in h:
        client, *servers = e["profile_ms"]                       # slice 0, then the server slices
        assert len(servers) == len(CUTS)
        assert {"forward", "backward", "optimizer", "send_fwd"} <= set(client)
        assert {"forward", "backward", "optimizer", "send_bwd"} <= set(servers[-1])
        assert all(v >= 0 for p in e["profile_ms"] for v in p.values())
        assert set(e["coordinator_ms"]) == {"data_load", "send", "wait"}
    monkeypatch.setattr(sys.modules[__name__], "_cfg", plain)
    _, _, h = _nodes("parallel", "copy", None)
    assert "profile_ms" not in h[0]          # without profiling the history has no such fields


# ── step 5: concurrent clients ────────────────────────────────────────────────────

def test_concurrent_one_client_at_a_time_is_the_client_by_client_session():
    # one client running at a time: client 0's whole round, then client 1's... each on
    # its own weights, as an in-process sequential session that hands nothing on
    ref, ref_states = _local("sequential", "keep", None)
    got, states, h = _nodes("parallel", "copy", None, client_schedule="concurrent",
                            max_concurrent_clients=1)
    assert [round(x, 6) for x in got] == ref
    _same_states(states, ref_states)
    assert "batches_skipped" not in h[0]


class _Timed(ts.WorkerServicer):
    """Records each round's (start, end) and stretches every batch a little, so rounds that
    run at once overlap for sure."""
    spans: list = []

    def client_round(self, request, context):
        import time
        t0 = time.monotonic()
        try:
            return super().client_round(request, context)
        finally:
            _Timed.spans.append((t0, time.monotonic()))

    def _drive_batch(self, batch_id, inputs, labels):
        import time
        time.sleep(0.02)
        super()._drive_batch(batch_id, inputs, labels)


@pytest.mark.parametrize("n_micro", [1, 2])
def test_concurrent_clients_run_at_once_and_aggregate(n_micro):
    _Timed.spans = []
    got, states, h = _nodes("parallel", "copy", ts.SplitFedAlgorithm(), n_micro=n_micro,
                            servicers={k: _Timed for k in range(C)}, client_schedule="concurrent")
    assert len(got) == 2 and all(0 < x < 10 for x in got)
    spans = sorted(_Timed.spans)
    for r in range(2):                                   # each round: every client was running
        rnd = spans[r * C:(r + 1) * C]                   # at the same moment
        assert max(a for a, _ in rnd) < min(b for _, b in rnd)
    _same_states([states[0], states[0]], [states[1], states[2]])    # FedAvg: one aggregate
    assert "batches_skipped" not in h[-1] and h[-1]["bytes_label"] > 0


def _dies_in_round(epoch):
    class _Dies(ts.WorkerServicer):
        def client_round(self, request, context):
            if request.epoch == epoch:
                self._server.stop(0)
                raise RuntimeError("client node killed")
            return super().client_round(request, context)
    return _Dies


def test_concurrent_a_dead_client_leaves_and_the_others_go_on():
    got, states, h = _nodes("parallel", "copy", ts.SplitFedAlgorithm(), client_schedule="concurrent",
                            servicers={1: _dies_in_round(1)})
    assert len(got) == 2 and h[0]["clients_lost"] == [1] and h[1]["clients_lost"] == [1]
    n1 = len(_loaders()[1])
    assert h[0]["batches_skipped"] == n1 and h[1]["batches_skipped"] == n1
    _same_states([states[0]], [states[2]])


def test_concurrent_refuses_a_sequential_session():
    with pytest.raises(ValueError, match="client_mode parallel"):
        _nodes("sequential", "copy", None, client_schedule="concurrent")


@pytest.mark.parametrize("sl,match", [
    ({"round_deadline_s": 1.0}, "round_deadline_s"),
    ({"client_schedule": "fast"}, "lockstep' or 'concurrent"),
])
def test_concurrent_refuses_what_it_cant_run(sl, match):
    cfg = _cfg()
    cfg.sl.client_nodes, cfg.sl.client_schedule = True, "concurrent"
    for k, v in sl.items():
        setattr(cfg.sl, k, v)
    with pytest.raises(ValueError, match=match):
        ts.ClientNodesExecutor(discovery=StaticDiscovery(peers=[]), coordinator_addr="127.0.0.1:1",
                               run_config=cfg)


class _ServerOrder(ts.WorkerServicer):
    """The batches the last slice stepped on, in the order it stepped (once per batch)."""
    order: list = []

    def _optimize(self, batch_id):
        if self.is_last:
            _ServerOrder.order.append(batch_id // self._n_micro)
        return super()._optimize(batch_id)


class _Replay(ts.ClientSession):
    """A parallel session whose rounds take the clients' batches in a given order."""
    def __init__(self, loaders, rounds):
        super().__init__(loaders, mode="parallel")
        self._rounds = iter(rounds)

    def _order(self, clients):
        its = {k: iter(self.loaders[k]) for k in clients}
        for k in next(self._rounds):
            yield k, next(its[k])


@pytest.mark.parametrize("n_micro", [1, 2])
def test_concurrent_equals_the_session_in_the_servers_order(n_micro):
    # what the server ran, replayed in one process, gives the same losses and weights: the
    # concurrent schedule changes the order of the server's updates and nothing else
    _ServerOrder.order, _Timed.spans = [], []
    got, states, _ = _nodes("parallel", "copy", ts.SplitFedAlgorithm(), n_micro=n_micro,
                            servicers={k: _Timed for k in range(C)}, server_servicer=_ServerOrder,
                            client_schedule="concurrent")
    per_round = sum(len(l) for l in _loaders())
    batches = list(_ServerOrder.order)
    rounds = [[b % C for b in batches[r * per_round:(r + 1) * per_round]] for r in range(2)]
    assert len(batches) == 2 * per_round
    assert any(r != sorted(r) for r in rounds)          # the clients really interleaved
    session = _Replay(_loaders(), rounds)
    ref = ts.slice(_model(), cuts=CUTS).train(session, run_config=_cfg(), algorithm=ts.SplitFedAlgorithm(),
                                              n_micro_batches=n_micro)
    assert [round(x, 6) for x in got] == [e["loss"] for e in ref]
    _same_states(states, session.states)


def _test_set():
    g = torch.Generator().manual_seed(7)
    return [(torch.randn(16, 8, generator=g), torch.randint(0, 4, (16,), generator=g)) for _ in range(3)]


@pytest.mark.parametrize("algorithm", [None, ts.SplitFedAlgorithm()])
def test_client_nodes_evaluate_like_the_in_process_session(algorithm):
    # without aggregation every client holds its own slice 0: each is evaluated with the
    # server and reported per client; after FedAvg they share one, evaluated once
    _, _, ref = _local("parallel", "copy", algorithm, eval_data=_test_set(), history=True)
    _, _, h = _nodes("parallel", "copy", algorithm, eval_data=_test_set())
    for r, g in zip(ref, h):
        for key in ("test_loss", "test_accuracy", "test_samples"):
            assert g[key] == pytest.approx(r[key], abs=1e-6), key
        assert ("test_loss_clients" in g) == ("test_loss_clients" in r) == (algorithm is None)
        if algorithm is None:
            assert g["test_loss_clients"] == pytest.approx(r["test_loss_clients"], abs=1e-6)
        assert g["bytes_eval"] > 0


def test_concurrent_client_nodes_evaluate():
    _, _, h = _nodes("parallel", "copy", ts.SplitFedAlgorithm(), eval_data=_test_set(),
                     client_schedule="concurrent")
    assert all(0 <= e["test_accuracy"] <= 1 and e["test_samples"] == 48 for e in h)
