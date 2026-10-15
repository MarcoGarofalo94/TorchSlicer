"""Run anywhere (M3): one script (``ts.run``) and one config give the same training in
one process, as OS processes, and in the rendered compose / k8s / bare-metal
deployments. Every mode that can run here must match unsplit PyTorch."""
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

import pytest
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

import torchslicer as ts
from torchslicer.deploy import plan_nodes, resolve_role
from torchslicer.deploy.launch import render_baremetal, render_compose, render_k8s
from torchslicer.deploy.nodes import node_names

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = "examples/run_anywhere/train.py"
SGD = {"name": "SGD", "params": {"lr": 0.05, "momentum": 0.9}}


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(),
                         nn.Linear(16, 4))


def _loader():
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(64, 8, generator=g), torch.randint(0, 4, (64,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)


def _reference(model_fn, loader_fn, epochs):
    model = model_fn()
    opt = torch.optim.SGD(model.parameters(), **SGD["params"])
    out = []
    for _ in range(epochs):
        losses = []
        for x, y in loader_fn():
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        out.append(sum(losses) / len(losses))
    return out


def _cfg(topology, n=3, transport="grpc"):
    cfg = ts.RunConfig()
    cfg.training.epochs = 2
    cfg.training.optimizer = SGD
    cfg.discovery.n_workers = n
    cfg.discovery.timeout = 20
    cfg.logging.enabled = False
    cfg.network.device = "cpu"
    cfg.transport.tensor = transport
    cfg.deploy.topology = topology
    return cfg


# ── roles ──────────────────────────────────────────────────────────────────────

def test_roles_are_validated(monkeypatch):
    monkeypatch.delenv("IS_DRIVER", raising=False)
    cfg = _cfg("centralized")
    assert resolve_role(cfg) == ("centralized", "all")
    cfg.deploy.role = "driver"
    with pytest.raises(ValueError, match="not valid for topology"):
        resolve_role(cfg)
    cfg.deploy.topology = "mesh"
    with pytest.raises(ValueError, match="deploy.topology"):
        resolve_role(cfg)
    # the M2 P2P scripts set IS_DRIVER, not ROLE
    cfg = _cfg("p2p")
    monkeypatch.setenv("IS_DRIVER", "true")
    assert resolve_role(cfg) == ("p2p", "driver")
    monkeypatch.setenv("IS_DRIVER", "false")
    assert resolve_role(cfg) == ("p2p", "follower")


def test_role_and_topology_come_from_env(monkeypatch):
    monkeypatch.setenv("TOPOLOGY", "P2P")
    monkeypatch.setenv("ROLE", "follower")
    cfg = ts.RunConfig.load()
    assert (cfg.deploy.topology, cfg.deploy.role) == ("p2p", "follower")


# ── one process: ROLE=all ──────────────────────────────────────────────────────

@pytest.mark.parametrize("topology,transport", [
    ("local", "grpc"), ("centralized", "grpc"), ("centralized", "tcp"),
    ("p2p", "grpc"), ("p2p", "tcp"),
])
def test_run_in_one_process_matches_unsplit(topology, transport):
    ref = _reference(_model, _loader, 2)
    h = ts.run(_model, _loader, run_config=_cfg(topology, transport=transport), verbose=False)
    assert [e["loss"] for e in h] == pytest.approx(ref, abs=1e-5)


def test_local_topology_honours_network_device():
    """``network.device`` means the same on every target; local used to ignore it and
    train wherever the model was built (CPU), while workers used CUDA when present."""
    seen = []

    def model_fn():
        m = _model()
        orig = m.to
        m.to = lambda *a, **k: (seen.append(a[0] if a else k.get("device")), orig(*a, **k))[1]
        return m

    ts.run(model_fn, _loader, run_config=_cfg("local"), verbose=False)
    assert seen and str(seen[0]) == "cpu"


def test_start_worker_honours_config_device(monkeypatch):
    """start_worker(run_config=) used the DEVICE env var only, ignoring network.device."""
    monkeypatch.setenv("DEVICE", "cuda")
    cfg = _cfg("centralized")            # network.device = cpu
    w = ts.start_worker(host="127.0.0.1", run_config=cfg)
    try:
        assert w.servicer.device.type == "cpu"
    finally:
        w.stop()


# ── start-up order: static peers may come up after the driver ──────────────────

def test_late_followers_are_waited_for_and_planned_with_their_capabilities():
    """Static discovery lists addresses, not live workers. The driver used to probe
    followers that were not up yet, warn, and plan without their capabilities
    (compute_gflops=0), so the plan depended on start-up timing."""
    cfg = _cfg("p2p")
    ports = [ts.deploy.free_port() for _ in range(2)]
    followers = []

    def start_late():
        time.sleep(1.5)
        followers.extend(ts.start_worker(p, host="127.0.0.1", run_config=cfg) for p in ports)

    t = threading.Thread(target=start_late)
    t.start()
    driver = f"127.0.0.1:{ts.deploy.free_port()}"
    ex = ts.P2PExecutor(peers=[f"127.0.0.1:{p}" for p in ports], address=driver,
                        bind_addr=driver, run_config=cfg)
    try:
        h = ts.slice(_model(), n=3, executor=ex).train(_loader(), run_config=cfg)
        assert [e["loss"] for e in h] == pytest.approx(_reference(_model, _loader, 2), abs=1e-5)
        assert all(nd.compute_gflops > 0 for nd in ex._nodes)
    finally:
        t.join()
        for w in followers:
            w.stop()


def test_a_follower_that_never_comes_up_fails_within_discovery_timeout():
    cfg = _cfg("p2p", n=2)
    cfg.discovery.timeout = 1
    driver = f"127.0.0.1:{ts.deploy.free_port()}"
    ex = ts.P2PExecutor(peers=[f"127.0.0.1:{ts.deploy.free_port()}"], address=driver,
                        bind_addr=driver, run_config=cfg)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="not reachable within discovery.timeout"):
        ts.slice(_model(), n=2, executor=ex).train(_loader(), run_config=cfg)
    assert time.monotonic() - t0 < 15
    ex.teardown()


# ── one OS process per node (the launcher's `processes` target) ────────────────

def _example_reference():
    sys.path.insert(0, str(ROOT / "examples" / "run_anywhere"))
    try:
        import train
        return _reference(train.build_model, train.get_loader, 2)
    finally:
        sys.path.pop(0)


def _launch(*args, env=None):
    r = subprocess.run([sys.executable, "-m", "torchslicer", "launch", SCRIPT, *args],
                       cwd=ROOT, capture_output=True, text=True, timeout=240,
                       env={**os.environ, **(env or {})})
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    line = next(l for l in r.stdout.splitlines() if l.startswith("HISTORY "))
    return json.loads(line[len("HISTORY "):])


@pytest.mark.parametrize("config,env", [
    ("centralized", {}),
    ("centralized", {"DISCOVERY_BACKEND": "static", "TENSOR_TRANSPORT": "tcp"}),
    ("p2p", {}),
])
def test_launch_processes_matches_unsplit(config, env, tmp_path):
    got = _launch("-c", f"experiments/run_anywhere_{config}.yaml", "--target", "processes",
                  "-o", str(tmp_path), env=env)
    assert got == pytest.approx(_example_reference(), abs=1e-5)


def test_launch_local_matches_unsplit():
    got = _launch("-c", "experiments/run_anywhere_p2p.yaml", "--target", "local")
    assert got == pytest.approx(_example_reference(), abs=1e-5)


# ── rendered targets: the same nodes, packaged ─────────────────────────────────

def _load(name):
    return ts.RunConfig.from_yaml(str(ROOT / "experiments" / f"run_anywhere_{name}.yaml"))


@pytest.mark.parametrize("name", ["centralized", "p2p"])
def test_node_plan_is_consistent(name):
    cfg = _load(name)
    names = node_names(cfg)
    nodes = plan_nodes(cfg, [n for n, _ in names], [50051 + i for i in range(len(names))],
                       discovery="static")
    lead, others = nodes[0], nodes[1:]
    assert len(names) == (4 if name == "centralized" else 3)
    assert lead.drives and not any(o.drives for o in others)
    assert len({nd.address for nd in nodes}) == len(nodes)
    peers = lead.env["WORKER_PEERS"].split(",")
    assert peers == ([o.address for o in others] if name == "centralized"
                     else [nd.address for nd in nodes])
    for nd in nodes:
        assert nd.env["ROLE"] == nd.role and nd.env["TOPOLOGY"] == name


def test_local_topology_has_nothing_to_deploy():
    with pytest.raises(ValueError, match="nothing to deploy"):
        node_names(_load("local"))


@pytest.mark.parametrize("name", ["centralized", "p2p"])
def test_render_compose(name, tmp_path):
    cfg = _load(name)
    doc = render_compose(cfg, SCRIPT, f"experiments/run_anywhere_{name}.yaml",
                         out=str(tmp_path / "out"), project_dir=str(ROOT))
    doc = yaml.safe_load(yaml.safe_dump(doc))
    svcs = doc["services"]
    lead = "coordinator" if name == "centralized" else "driver"
    assert list(svcs)[0] == lead and len(svcs) == len(node_names(cfg))
    assert set(svcs[lead]["depends_on"]) == set(svcs) - {lead}
    for sname, s in svcs.items():
        assert s["hostname"] == sname and s["command"] == ["python3", SCRIPT]
        assert s["environment"]["EXPERIMENT_CONFIG"] == f"experiments/run_anywhere_{name}.yaml"
        assert s["volumes"][0].endswith(":/workspace")
    peers = svcs[lead]["environment"].get("WORKER_PEERS")
    if peers:   # static discovery: every peer is a service on the compose network
        assert {p.split(":")[0] for p in peers.split(",")} <= set(svcs)


@pytest.mark.parametrize("name,transport", [("centralized", "grpc"), ("p2p", "tcp")])
def test_render_k8s(name, transport):
    cfg = _load(name)
    cfg.transport.tensor = transport
    cfg.deploy.gpus = 1
    text = (ROOT / "experiments" / f"run_anywhere_{name}.yaml").read_text()
    docs = list(yaml.safe_load_all(yaml.safe_dump_all(render_k8s(cfg, SCRIPT, text))))
    kinds = {(d["kind"], d["metadata"]["name"]): d for d in docs}
    lead = "coordinator" if name == "centralized" else "driver"
    pre = cfg.deploy.name
    cm = kinds[("ConfigMap", f"{pre}-config")]
    assert cm["data"]["run.yaml"] == text
    wjob = kinds[("Job", f"{pre}-workers")]
    k = len(node_names(cfg)) - 1
    assert wjob["spec"]["completionMode"] == "Indexed"
    assert wjob["spec"]["completions"] == wjob["spec"]["parallelism"] == k
    wsvc = kinds[("Service", f"{pre}-workers")]
    assert wsvc["spec"]["clusterIP"] == "None"
    wpod = wjob["spec"]["template"]
    assert wpod["spec"]["subdomain"] == f"{pre}-workers"
    assert wsvc["spec"]["selector"].items() <= wpod["metadata"]["labels"].items()
    ljob = kinds[("Job", f"{pre}-{lead}")]
    lsvc = kinds[("Service", f"{pre}-{lead}")]
    assert lsvc["spec"]["selector"].items() <= ljob["spec"]["template"]["metadata"]["labels"].items()
    if transport == "tcp":
        assert [p["name"] for p in lsvc["spec"]["ports"]] == ["grpc", "tensor"]
    env = {e["name"]: e["value"] for e in ljob["spec"]["template"]["spec"]["containers"][0]["env"]}
    port = cfg.network.worker_port
    workers = [f"{pre}-workers-{i}.{pre}-workers:{port}" for i in range(k)]
    # Indexed Job pod i has hostname <job>-<i>; with the subdomain it resolves as
    # <job>-<i>.<service>, so the static peer list is the pod order.
    assert env["WORKER_PEERS"].split(",") == (
        workers if name == "centralized" else [f"{pre}-{lead}:{port}", *workers])
    assert env["EXPERIMENT_CONFIG"] == "/etc/torchslicer/run.yaml"
    c = ljob["spec"]["template"]["spec"]["containers"][0]
    assert c["resources"]["limits"]["nvidia.com/gpu"] == 1
    for job in (wjob, ljob):
        assert job["spec"]["template"]["spec"]["restartPolicy"] == "Never"


def test_render_baremetal_gives_nodes_on_one_host_distinct_ports(tmp_path):
    cfg = _load("p2p")
    cfg.transport.tensor = "tcp"
    paths = render_baremetal(cfg, SCRIPT, "experiments/run_anywhere_p2p.yaml",
                             ["h0", "h1", "h1"], out=str(tmp_path), workdir="/srv/run")
    env = {}
    for p in paths:
        for line in pathlib.Path(p).read_text().splitlines():
            if line.startswith("export WORKER_PEERS="):
                env["peers"] = line.split("=", 1)[1]
    assert env["peers"] == "h0:50051,h1:50051,h1:50053"   # tcp: port + 1 is taken
    with pytest.raises(SystemExit, match="needs 3 hosts"):
        render_baremetal(cfg, SCRIPT, "x.yaml", ["h0"], out=str(tmp_path), workdir="/")
