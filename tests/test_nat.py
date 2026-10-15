"""NAT traversal (M6): a node behind NAT opens no port and only dials out; calls to it
come back over its tunnels. With a relay, every node dials only the relay. Training
must match unsplit PyTorch and the direct run (same losses, same bytes per cut) in
every topology and setting, and a NAT'd node must never accept a connection.
"""
import time

import pytest
import torch
import torch.nn as nn

import torchslicer as ts
from torchslicer.transport.tunnel import Router, TunnelRpcError, start_relay

SGD = {"name": "SGD", "params": {"lr": 0.05, "momentum": 0.9}}
CE = {"name": "CrossEntropyLoss", "params": {}}
CUTS = ["1", "3"]          # 3 slices: Linear+ReLU | Linear+ReLU | Linear
B = 16
METRICS = ("bytes_fwd", "bytes_bwd", "bytes_input", "bytes_label", "bytes_total", "bytes_client")


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))


def _data():
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(64, 8, generator=g), torch.randint(0, 4, (64,), generator=g)
    return [(x[i:i + B], y[i:i + B]) for i in range(0, 64, B)]


def _reference(epochs=2):
    model = _model()
    opt = torch.optim.SGD(model.parameters(), **SGD["params"])
    out = []
    for _ in range(epochs):
        losses = []
        for x, y in _data():
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        out.append(sum(losses) / len(losses))
    return out


def _cfg(topology="p2p", setting="", transport="grpc", nat=False, relay=""):
    cfg = ts.RunConfig()
    cfg.training.epochs = 2
    cfg.training.optimizer = SGD
    cfg.training.criterion = CE
    cfg.discovery.n_workers = 3
    cfg.discovery.timeout = 20
    cfg.logging.enabled = False
    cfg.network.device = "cpu"
    cfg.transport.tensor = transport
    cfg.transport.nat = nat
    cfg.transport.relay = relay
    cfg.deploy.topology = topology
    cfg.planning.heterogeneous = False
    cfg.sl.setting = setting
    cfg.fault_tolerance.heartbeat_interval_s = 0.2
    cfg.fault_tolerance.ping_timeout_s = 1.0
    return cfg


def _run(cfg, **kw):
    return ts.run(_model, _data, run_config=cfg, verbose=False, **{"cuts": CUTS, **kw})


def _losses(history):
    return [h["loss"] for h in history]


# ── the data owner behind NAT (ROLE=all: workers on loopback) ──────────────────

@pytest.mark.parametrize("topology,setting,transport", [
    ("p2p", "vanilla", "grpc"),
    ("p2p", "vanilla", "tcp"),
    ("p2p", "u_shaped", "tcp"),
    ("centralized", "vanilla", "grpc"),
    ("centralized", "u_shaped", "tcp"),
])
def test_data_owner_behind_nat_matches_direct(topology, setting, transport):
    direct = _run(_cfg(topology, setting, transport))
    got = _run(_cfg(topology, setting, transport, nat=True))
    assert _losses(got) == pytest.approx(_reference(), abs=1e-6)
    assert _losses(got) == _losses(direct)
    for k in METRICS:
        assert got[0][k] == direct[0][k], k


@pytest.mark.parametrize("topology,setting,transport", [
    ("p2p", "u_shaped", "grpc"),
    ("centralized", "pipeline", "tcp"),
    ("centralized", "vanilla", "grpc"),
])
def test_everything_through_a_relay_matches_direct(topology, setting, transport):
    direct = _run(_cfg(topology, setting, transport))
    got = _run(_cfg(topology, setting, transport, nat=True, relay="auto"))
    assert _losses(got) == pytest.approx(_reference(), abs=1e-6)
    for k in METRICS:
        assert got[0][k] == direct[0][k], k


# ── what crosses which connection ──────────────────────────────────────────────

def _followers(cfg, k, **kw):
    return [ts.start_worker(ts.deploy.runner.free_port(1), host="127.0.0.1", run_config=cfg, **kw)
            for _ in range(k)]


def _p2p(cfg, followers):
    addr = f"127.0.0.1:{ts.deploy.runner.free_port(1)}"
    return ts.P2PExecutor(peers=[w.address for w in followers], address=addr, bind_addr=addr,
                          run_config=cfg)


def test_nat_driver_listens_on_nothing_and_gets_its_calls_by_tunnel():
    """The driver opens no port; the gradients for slice 0 (one per batch) and the
    losses (one per batch) reach it over the tunnels it opened."""
    cfg = _cfg(setting="vanilla", transport="tcp")
    followers = _followers(cfg, 2)
    cfg.transport.nat = True
    try:
        ex = _p2p(cfg, followers)
        h = ts.slice(_model(), cuts=CUTS, executor=ex).train(_data(), run_config=cfg)
        assert ex._grpc_server is None and ex._tail_server is None
        assert ex.local._tensor_server is None
        assert ex._router.stats["calls_in"] >= 2 * 4 * 2        # backward + report_metrics
        assert ex._router.kept() == []                           # closed at teardown
        # the follower next to the driver saw it only through the tunnel
        assert followers[0].servicer.router.stats["calls_out"] >= 2 * 4
    finally:
        for w in followers:
            w.stop()
    assert _losses(h) == pytest.approx(_reference(), abs=1e-6)


def test_nat_coordinator_cannot_take_registrations():
    cfg = _cfg("centralized", "vanilla", nat=True)
    ex = ts.DistributedExecutor(ts.CoordinatorDiscovery(run_id="x"), coordinator_addr="127.0.0.1:1",
                                run_config=cfg)
    with pytest.raises(ValueError, match="can't register"):
        ts.slice(_model(), cuts=CUTS, executor=ex).train(_data(), run_config=cfg)


class Tied(nn.Module):      # module level: slices are pickled to the workers
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(10, 8)
        self.mid = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
        self.head = nn.Linear(8, 10, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, x):
        return self.head(self.mid(self.emb(x)))


def test_tied_weights_across_a_nat_driver():
    """The embedding (driver) and the head (last follower) share a weight: the
    follower's gradient for it reaches the driver by tunnel, and the update goes back."""
    g = torch.Generator().manual_seed(3)
    data = [(torch.randint(0, 10, (8,), generator=g), torch.randint(0, 10, (8,), generator=g))
            for _ in range(4)]
    local = ts.slice(Tied(), cuts=["emb", "mid"]).train(data, SGD, CE, epochs=2,
                                                        run_config=_cfg("local"))
    cfg = _cfg(setting="vanilla")
    followers = _followers(cfg, 2)
    cfg.transport.nat = True
    try:
        ex = _p2p(cfg, followers)
        got = ts.slice(Tied(), cuts=["emb", "mid"], executor=ex).train(data, run_config=cfg)
    finally:
        for w in followers:
            w.stop()
    assert _losses(got) == pytest.approx(_losses(local), abs=1e-6)


# ── workers behind NAT ─────────────────────────────────────────────────────────

def _coordinator(cfg, discovery):
    port = ts.deploy.runner.free_port()
    addr = f"127.0.0.1:{port}"
    return ts.DistributedExecutor(discovery, coordinator_addr=addr, coordinator_bind_addr=addr,
                                  run_config=cfg), addr


@pytest.mark.parametrize("transport", ["grpc", "tcp"])
def test_nat_workers_register_and_train(transport):
    """Workers 0 and 2 of a centralized pipeline are behind NAT, worker 1 is not: they
    register with the coordinator, open tunnels to it and to worker 1, and train."""
    cfg = _cfg("centralized", "pipeline", transport)
    ex, addr = _coordinator(cfg, ts.CoordinatorDiscovery(run_id="nat"))
    workers = []
    try:
        import threading
        out = {}
        t = threading.Thread(target=lambda: out.setdefault("h", ts.slice(
            _model(), cuts=CUTS, executor=ex).train(_data(), run_config=cfg)))
        t.start()
        from torchslicer.discovery import announce_to_coordinator, NodeInfo
        for i, nat in enumerate([True, False, True]):
            w = (ts.start_worker(host="127.0.0.1", address=f"nat-worker-{i}:1", run_config=cfg,
                                 nat=True, tunnel_to=[addr]) if nat else
                 ts.start_worker(ts.deploy.runner.free_port(1), host="127.0.0.1", run_config=cfg))
            workers.append(w)
            if nat:
                assert w.servicer.router.keep(addr).wait(10)
            announce_to_coordinator(addr, NodeInfo(node_id=w.address, address=w.address),
                                    channel=w.servicer.router.channel(addr))
            time.sleep(0.2)   # registration order = slice order
        t.join(60)
        assert not t.is_alive()
    finally:
        for w in workers:
            w.stop()
    assert _losses(out["h"]) == pytest.approx(_reference(), abs=1e-6)
    assert workers[0].servicer._tensor_server is None and workers[2].servicer._tensor_server is None
    assert workers[1].servicer.router.stats["calls_out"] >= 2 * 4 * 2   # fwd to 2, bwd to 0


def test_adjacent_nat_workers_need_a_relay():
    cfg = _cfg("p2p", "vanilla")
    ex = None
    addr = f"127.0.0.1:{ts.deploy.runner.free_port(1)}"
    w = [ts.start_worker(host="127.0.0.1", address=f"nat-{i}:1", run_config=cfg, nat=True,
                         tunnel_to=[addr]) for i in range(2)]
    try:
        ex = ts.P2PExecutor(peers=[x.address for x in w], address=addr, bind_addr=addr,
                            run_config=cfg)
        with pytest.raises(RuntimeError, match="through a relay"):
            ts.slice(_model(), cuts=CUTS, executor=ex).train(_data(), run_config=cfg)
    finally:
        for x in w:
            x.stop()
        if ex is not None:
            ex.teardown()


def test_every_node_behind_nat_through_a_relay():
    """Driver and both followers behind NAT (the followers adjacent): only the relay
    accepts connections."""
    relay = start_relay(host="127.0.0.1")
    cfg = _cfg(setting="u_shaped", nat=True, relay=relay.address)
    cfg.discovery.n_workers = 4
    w = [ts.start_worker(host="127.0.0.1", address=f"nat-{i}:1", run_config=cfg) for i in range(2)]
    try:
        assert all(x.servicer.nat for x in w)
        addr = "nat-driver:1"
        ex = ts.P2PExecutor(peers=[x.address for x in w], address=addr, run_config=cfg)
        h = ts.slice(_model(), cuts=["1", "2", "3"], executor=ex).train(_data(), run_config=cfg)
    finally:
        for x in w:
            x.stop()
        relay.stop()
    assert _losses(h) == pytest.approx(_reference(), abs=1e-6)
    assert relay.router.stats["forwarded"] > 0


# ── failures ───────────────────────────────────────────────────────────────────

def test_dead_follower_behind_nat_fails_the_run_instead_of_hanging():
    """A NAT'd follower that dies closes its tunnel; the driver can't dial it, so the
    next heartbeat fails at once (UNAVAILABLE), as with a direct connection."""
    cfg = _cfg(setting="vanilla")
    addr = f"127.0.0.1:{ts.deploy.runner.free_port(1)}"
    w = [ts.start_worker(host="127.0.0.1", address=f"nat-{i}:1", run_config=cfg, nat=True,
                         tunnel_to=[addr]) for i in range(1)] + _followers(cfg, 1)
    w.reverse()   # slice 1 reachable, slice 2 behind NAT
    calls = {"n": 0}

    def data():
        for x, y in _data() * 50:
            calls["n"] += 1
            if calls["n"] == 3:
                w[1].stop()
            yield x, y

    class Loader:
        def __iter__(self):
            return data()

        def __len__(self):
            return 200

    try:
        ex = ts.P2PExecutor(peers=[x.address for x in w], address=addr, bind_addr=addr,
                            run_config=cfg)
        t0 = time.time()
        with pytest.raises(ts.WorkerFailureError):
            ts.slice(_model(), cuts=CUTS, executor=ex).train(Loader(), run_config=cfg)
        assert time.time() - t0 < 30
    finally:
        for x in w:
            x.stop()


class _StopAfter(ts.TrainingCallback):
    def __init__(self, worker, epoch):
        self.worker, self.epoch = worker, epoch

    def on_epoch_end(self, epoch, metrics):
        if epoch == self.epoch:
            self.worker.stop()
        return metrics


def test_nat_driver_recovers_from_a_dead_follower(tmp_path):
    """Recovery re-plans over the survivors; the driver's tunnel to the dead follower is
    released and the survivors keep calling it back over theirs."""
    cfg = _cfg(setting="vanilla")
    cfg.training.epochs = 3
    cfg.fault_tolerance.enabled, cfg.fault_tolerance.max_retries = True, 1
    cfg.checkpoint.dir = str(tmp_path)
    followers = _followers(cfg, 2)
    cfg.transport.nat = True
    try:
        ex = _p2p(cfg, followers)
        h = ts.slice(_model(), n=3, executor=ex).train(
            _data(), run_config=cfg, callbacks=[_StopAfter(followers[1], 1)])
        assert len(h) == 3 and all(e["loss"] == e["loss"] for e in h)
        assert [nd.address for nd in ex._nodes] == [ex.address, followers[0].address]
        assert ex._grpc_server is None
    finally:
        for w in followers:
            w.stop()


# ── tunnel unit ────────────────────────────────────────────────────────────────

def test_tunnel_errors_look_like_grpc_errors():
    import grpc
    server_cfg = _cfg()
    worker = _followers(server_cfg, 1)[0]
    router = Router()
    stub_addr = worker.address

    class Echo(ts.WorkerServicer):
        def echo(self, request, context):
            if request.payload == b"boom":
                raise RuntimeError("boom")
            if request.payload == b"slow":
                time.sleep(1.0)
            return super().echo(request, context)

    from torchslicer.transport.grpc.worker import worker_service_pb2 as pb, worker_service_pb2_grpc as pbg
    router.serve("nat-x:1", Echo(), pbg.add_WorkerServiceServicer_to_server)
    try:
        router.keep(stub_addr, wait=10)
        stub = pbg.WorkerServiceStub(worker.servicer.router.channel("nat-x:1"))
        assert stub.echo(pb.EchoRequest(payload=b"abc")).n_bytes == 3
        with pytest.raises(grpc.RpcError) as e:
            stub.echo(pb.EchoRequest(payload=b"boom"))
        assert e.value.code() == grpc.StatusCode.UNKNOWN and "boom" in e.value.details()
        with pytest.raises(grpc.RpcError) as e:
            stub.echo(pb.EchoRequest(payload=b"slow"), timeout=0.2)
        assert e.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
        router.release(stub_addr)
        time.sleep(0.3)
        with pytest.raises(grpc.RpcError) as e:      # tunnel gone: fails fast, never dials
            stub.echo(pb.EchoRequest(payload=b"abc"))
        assert e.value.code() == grpc.StatusCode.UNAVAILABLE
        assert isinstance(e.value, TunnelRpcError)
    finally:
        router.close()
        worker.stop()


def test_relay_auto_is_only_for_the_launcher():
    with pytest.raises(ValueError, match="relay's address"):
        Router("auto")


# ── launcher: real OS processes, and rendered targets ──────────────────────────

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
NAT_CFG = "experiments/nat_u_shaped_p2p.yaml"


def _launch(*args, env=None):
    r = subprocess.run([sys.executable, "-m", "torchslicer", "launch",
                        "examples/run_anywhere/train.py", *args],
                       cwd=ROOT, capture_output=True, text=True, timeout=240,
                       env={**os.environ, **(env or {})})
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    line = next(l for l in r.stdout.splitlines() if l.startswith("HISTORY "))
    return json.loads(line[len("HISTORY "):]), r.stdout


@pytest.mark.parametrize("env", [
    {"TENSOR_TRANSPORT": "tcp"},                                       # driver behind NAT
    {"TRANSPORT_RELAY": "auto"},                                       # + a relay process
    {"TOPOLOGY": "centralized", "SL_SETTING": "vanilla", "DISCOVERY_BACKEND": "static",
     "TRANSPORT_RELAY": "auto"},
])
def test_launch_processes_behind_nat_matches_local(env, tmp_path):
    local, _ = _launch("-c", "experiments/run_anywhere_p2p.yaml", "--target", "local",
                       env={"SL_SETTING": "u_shaped"})
    got, out = _launch("-c", NAT_CFG, "--target", "processes", "-o", str(tmp_path), env=env)
    assert got == pytest.approx(local, abs=1e-6)
    assert ("relay " in out) == ("TRANSPORT_RELAY" in env)
    if "TRANSPORT_RELAY" in env:
        assert "tunnel opened" in (tmp_path / "relay.log").read_text()


def test_plan_marks_the_data_owner_behind_nat_and_adds_a_relay():
    from torchslicer.deploy.nodes import plan_nodes, node_names
    cfg = ts.RunConfig.load(str(ROOT / NAT_CFG))
    names = node_names(cfg)
    nodes = plan_nodes(cfg, ["h"] * len(names), list(range(1, len(names) + 1)))
    assert [(n.role, n.env.get("TRANSPORT_NAT")) for n in nodes] == [("driver", "true"),
                                                                     ("follower", "false")]
    assert nodes[1].env["COORDINATOR_ADDRESS"] == nodes[0].address
    cfg.transport.relay = "auto"
    names = node_names(cfg)
    assert names[-1] == ("relay", "relay")
    nodes = plan_nodes(cfg, ["a", "b", "r"], [1, 2, 3])
    assert all(n.env["TRANSPORT_RELAY"] == "r:3" for n in nodes[:-1])
    assert nodes[-1].argv("x.py") == ["-m", "torchslicer", "relay", "--port", "3"]
    cfg.deploy.topology, cfg.sl.setting, cfg.transport.relay = "centralized", "vanilla", ""
    with pytest.raises(ValueError, match="can't register"):
        plan_nodes(cfg, ["a", "b", "c"], [1, 2, 3])


def test_k8s_and_compose_render_the_relay():
    from torchslicer.deploy.launch import render_compose, render_k8s
    cfg = ts.RunConfig.load(str(ROOT / NAT_CFG))
    cfg.transport.relay = "auto"
    docs = render_k8s(cfg, "train.py", "x: 1")
    dep = next(d for d in docs if d["kind"] == "Deployment")
    assert dep["metadata"]["name"] == "torchslicer-relay"
    assert dep["spec"]["template"]["spec"]["containers"][0]["command"][-4:] == \
        ["torchslicer", "relay", "--port", "50060"]
    jobs = [d for d in docs if d["kind"] == "Job"]
    envs = [{e["name"]: e["value"] for e in j["spec"]["template"]["spec"]["containers"][0]["env"]}
            for j in jobs]
    assert all(e["TRANSPORT_RELAY"] == "torchslicer-relay:50060" for e in envs)
    compose = render_compose(cfg, "train.py", "c.yaml")
    assert compose["services"]["relay"]["command"][-1] == "50060"
    assert "relay" in compose["services"]["driver"]["depends_on"]
