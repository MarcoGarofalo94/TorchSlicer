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


def _local(mode, handoff, algorithm, n_micro=1):
    session = ts.ClientSession(_loaders(), mode=mode, handoff=handoff)
    h = ts.slice(_model(), cuts=CUTS).train(session, run_config=_cfg(), algorithm=algorithm,
                                           n_micro_batches=n_micro)
    return [e["loss"] for e in h], session.states


def _nodes(mode, handoff, algorithm, transport="grpc", n_micro=1):
    cfg = _cfg(transport)
    offsets = cfg.transport.listen_offsets()
    servers = [ts.start_worker(free_port(offsets), host="127.0.0.1", run_config=cfg) for _ in range(2)]
    data = _loaders()
    clients = [ts.start_worker(free_port(offsets), host="127.0.0.1", run_config=cfg, data=data[k],
                               client_index=k) for k in range(C)]
    peers = [w.address for w in clients[::-1] + servers]      # discovery order doesn't matter
    addr = f"127.0.0.1:{free_port(cfg.transport.listen_offsets(drives=True))}"
    ex = ts.ClientNodesExecutor(discovery=StaticDiscovery(peers=peers), coordinator_addr=addr,
                                run_config=cfg)
    session = ts.RemoteSession(C, mode=mode, handoff=handoff)
    try:
        h = ts.slice(_model(), cuts=CUTS, executor=ex).train(session, run_config=cfg, algorithm=algorithm,
                                                            n_micro_batches=n_micro)
        states = session.states          # every client's final state, read at teardown
    finally:
        for w in servers + clients:
            w.stop()
    return [e["loss"] for e in h], states, h


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
