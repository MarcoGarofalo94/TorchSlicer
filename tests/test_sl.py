"""SL settings (M4): roles/placement, cut hooks, SL metrics and multi-client sessions.

Every setting and session must train exactly like the equivalent local run (and, without
lossy hooks, like unsplit PyTorch) in every topology; the byte metrics must be what the
cut tensors weigh, identical whether measured by LocalExecutor or by the workers.
"""
import json
import logging
import os
import pathlib
import subprocess
import sys
import threading
import time

import pytest
import torch
import torch.nn as nn

import torchslicer as ts
from torchslicer.deploy.launch import render_baremetal, render_k8s
from torchslicer.deploy.nodes import node_names, plan_nodes
from torchslicer.sl.hooks import pack_payload, payload_nbytes, unpack_payload, resolve_cut_hooks

ROOT = pathlib.Path(__file__).resolve().parent.parent
SGD = {"name": "SGD", "params": {"lr": 0.05, "momentum": 0.9}}
CE = {"name": "CrossEntropyLoss", "params": {}}
CUTS = ["1", "3"]          # 3 slices: Linear+ReLU | Linear+ReLU | Linear
B, D_IN, D_H = 16, 8, 16   # batch, input width, activation width at both cuts


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(D_IN, D_H), nn.ReLU(), nn.Linear(D_H, D_H), nn.ReLU(),
                         nn.Linear(D_H, 4))


def _data():
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(64, D_IN, generator=g), torch.randint(0, 4, (64,), generator=g)
    return [(x[i:i + B], y[i:i + B]) for i in range(0, 64, B)]


def _reference(batches, epochs=2):
    model = _model()
    opt = torch.optim.SGD(model.parameters(), **SGD["params"])
    out = []
    for _ in range(epochs):
        losses = []
        for x, y in batches:
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        out.append(sum(losses) / len(losses))
    return out


def _cfg(topology="local", setting="", transport="grpc", **sl):
    cfg = ts.RunConfig()
    cfg.training.epochs = 2
    cfg.training.optimizer = SGD
    cfg.training.criterion = CE
    cfg.discovery.n_workers = 3
    cfg.discovery.timeout = 20
    cfg.logging.enabled = False
    cfg.network.device = "cpu"
    cfg.transport.tensor = transport
    cfg.deploy.topology = topology
    cfg.planning.heterogeneous = False
    cfg.sl.setting = setting
    for k, v in sl.items():
        setattr(cfg.sl, k, v)
    return cfg


def _run(cfg, data_fn=_data, **kw):
    return ts.run(_model, data_fn, run_config=cfg, verbose=False, **{"cuts": CUTS, **kw})


def _losses(history):
    return [h["loss"] for h in history]


METRICS = ("bytes_fwd", "bytes_bwd", "bytes_input", "bytes_label", "bytes_total", "bytes_client",
           "raw_bytes_total")


# ── payloads ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("payload", [
    torch.randn(3, 4),
    [torch.randint(0, 255, (5,), dtype=torch.uint8), torch.tensor([1.5, 2.0]),
     torch.tensor([2, 3], dtype=torch.int64), torch.zeros(0), torch.ones(2, dtype=torch.bool)],
    torch.randn(2, 2).to(torch.bfloat16),
])
def test_payload_packing_round_trips_and_counts_its_bytes(payload):
    packed = pack_payload(payload)
    assert packed.dtype == torch.uint8 and packed.numel() == payload_nbytes(payload)
    back = unpack_payload(packed)
    if isinstance(payload, torch.Tensor):
        assert back.dtype == payload.dtype and torch.equal(back, payload)
    else:
        assert all(a.dtype == b.dtype and torch.equal(a, b) for a, b in zip(back, payload))


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
def test_quantize_error_is_within_half_a_step(bits):
    from torchslicer.sl.hooks import CutContext
    x = torch.randn(7, 13)
    h = ts.Quantize(bits=bits)
    ctx = CutContext(0, "forward", 0)
    payload = h.encode(x, ctx)
    step = (x.max() - x.min()) / (2 ** bits - 1)
    assert (h.decode(payload, ctx) - x).abs().max() <= step / 2 + 1e-6
    assert payload[0].numel() == -(-x.numel() * bits // 8)    # bit-packed


def test_topk_keeps_the_largest_entries_per_sample():
    from torchslicer.sl.hooks import CutContext
    x = torch.tensor([[0.1, -3.0, 0.2, 2.0], [5.0, 0.0, -0.5, 0.4]])
    ctx = CutContext(0, "forward", 0)
    h = ts.TopK(ratio=0.5)
    assert torch.equal(h.decode(h.encode(x, ctx), ctx),
                       torch.tensor([[0.0, -3.0, 0.0, 2.0], [5.0, 0.0, -0.5, 0.0]]))


def test_cut_hook_resolution():
    q, f = ts.Quantize(), ts.Float16()
    assert resolve_cut_hooks(q, 3) == {0: q, 1: q}
    assert resolve_cut_hooks({1: f}, 3) == {1: f}
    got = resolve_cut_hooks({0: f}, 3, [{"cut": "all", "name": "quantize", "params": {"bits": 4}}])
    assert got[0] is f and isinstance(got[1], ts.Quantize) and got[1].bits == 4
    assert isinstance(resolve_cut_hooks([ts.GaussianNoise(0.1), q], 2)[0], ts.Chain)
    with pytest.raises(ValueError, match="cuts 0..1"):
        resolve_cut_hooks({2: q}, 3)
    assert resolve_cut_hooks({2: q}, 3, drop_missing=True) == {}
    with pytest.raises(ValueError, match="unknown cut hook"):
        resolve_cut_hooks(None, 3, [{"name": "zip"}])


def test_gaussian_noise_leaves_the_global_rng_alone():
    from torchslicer.sl.hooks import CutContext
    torch.manual_seed(123)
    before = torch.get_rng_state()
    h = ts.GaussianNoise(0.5, seed=7)
    a = h.encode(torch.zeros(4), CutContext(0, "forward", 3))
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(a, ts.GaussianNoise(0.5, seed=7).encode(torch.zeros(4), CutContext(0, "forward", 3)))
    assert not torch.equal(a, h.encode(torch.zeros(4), CutContext(0, "forward", 4)))


# ── local: settings, hooks, metrics ─────────────────────────────────────────────

def test_local_metrics_count_the_cut_tensors():
    h = _run(_cfg(link_bandwidth_mbps=8.0, link_latency_ms=1.0))
    assert _losses(h) == pytest.approx(_reference(_data()), abs=1e-6)
    e = h[0]
    act = 4 * B * D_H * 4                      # 4 batches of float32 [16, 16] per cut
    assert e["bytes_fwd"] == [act, act] and e["bytes_bwd"] == [act, act]
    assert e["bytes_input"] == 0 and e["bytes_label"] == 4 * B * 8   # vanilla: int64 labels leave
    assert e["bytes_client"] == 2 * act + e["bytes_label"]           # cut 0 + labels
    assert e["bytes_total"] == e["raw_bytes_total"] == 4 * act + e["bytes_label"]
    assert e["bytes_per_batch"] == e["bytes_total"] / 4
    assert len(e["compute_s"]) == 3 and e["client_compute_s"] == e["compute_s"][0]
    # 8 Mbit/s = 1 byte/us; 1 ms per message (4 batches x (2 cuts x 2 directions + 1 label))
    assert e["sim_comm_s"] == pytest.approx(e["bytes_total"] / 1e6 + 20 * 1e-3)
    assert e["sim_client_comm_s"] == pytest.approx(e["bytes_client"] / 1e6 + 12 * 1e-3)


@pytest.mark.parametrize("setting,inputs,labels,client_cuts", [
    ("pipeline", True, True, []),
    ("vanilla", False, True, [0]),
    ("u_shaped", False, False, [0, 1]),
])
def test_settings_change_placement_not_training(setting, inputs, labels, client_cuts):
    h = _run(_cfg(setting=setting))
    assert _losses(h) == pytest.approx(_reference(_data()), abs=1e-6)
    e = h[0]
    act = 4 * B * D_H * 4
    assert e["sl_setting"] == setting
    assert e["bytes_input"] == (4 * B * D_IN * 4 if inputs else 0)
    assert e["bytes_label"] == (4 * B * 8 if labels else 0)
    assert e["bytes_client"] == 2 * act * len(client_cuts) + e["bytes_input"] + e["bytes_label"]


def test_u_shaped_needs_three_slices():
    cfg = _cfg(setting="u_shaped")
    cfg.discovery.n_workers = 2
    with pytest.raises(ValueError, match="at least 3 slices"):
        _run(cfg, cuts=["3"])


def test_identity_hook_is_bit_exact_and_lossy_hooks_cut_bytes():
    plain = _run(_cfg())
    assert _losses(_run(_cfg(), cut_hooks=ts.CutHook())) == _losses(plain)
    q = _run(_cfg(), cut_hooks={0: ts.Quantize(bits=8)})
    act = B * D_H * 4
    # per message: header + uint8 values + [min, scale] float32 + shape int64[2]
    per_msg = payload_nbytes([torch.zeros(B, D_H, dtype=torch.uint8), torch.zeros(2),
                              torch.zeros(2, dtype=torch.int64)])
    assert q[0]["bytes_fwd"] == [4 * per_msg, 4 * act]
    assert q[0]["bytes_bwd"] == [4 * per_msg, 4 * act]
    assert q[0]["raw_bytes_total"] == plain[0]["raw_bytes_total"]
    assert _losses(q) == pytest.approx(_losses(plain), abs=1e-3) and _losses(q) != _losses(plain)


def test_config_hooks_apply_like_python_ones():
    py = _run(_cfg(), cut_hooks={0: ts.Quantize(bits=4)})
    cf = _run(_cfg(cut_hooks=[{"cut": 0, "name": "quantize", "params": {"bits": 4}}]))
    assert _losses(py) == _losses(cf) and py[0]["bytes_fwd"] == cf[0]["bytes_fwd"]


def test_recorder_sees_what_the_receiver_gets():
    rec = ts.Recorder(max_batches=2)
    _run(_cfg(), cut_hooks={1: rec})
    fwd = [t for d, _, t in rec.records if d == "forward"]
    bwd = [t for d, _, t in rec.records if d == "backward"]
    assert len(fwd) == len(bwd) == 2 and fwd[0].shape == (B, D_H)


def test_gpipe_with_hooks_counts_micro_batches():
    h = _run(_cfg(), cut_hooks={0: ts.Float16()})
    g = ts.slice(_model(), cuts=CUTS, cut_hooks={0: ts.Float16()}).train(
        _data(), SGD, CE, epochs=2, use_gpipe=True, n_micro_batches=2, run_config=_cfg())
    assert g[0]["bytes_fwd"][1] == h[0]["bytes_fwd"][1]           # same tensors, split in two
    assert g[0]["bytes_fwd"][0] > h[0]["bytes_fwd"][0]            # one header per micro-batch


# ── local: multi-client sessions ───────────────────────────────────────────────

def test_sequential_session_with_copy_handoff_is_training_on_the_concatenation():
    data = _data()
    session = ts.ClientSession([data[:1], data[1:3], data[3:]], mode="sequential")
    h = _run(_cfg(), data_fn=lambda: session)
    assert _losses(h) == _losses(_run(_cfg()))
    assert all(s is not None for s in session.states)


def test_parallel_session_of_one_client_is_plain_training():
    h = _run(_cfg(), data_fn=lambda: ts.ClientSession([_data()], mode="parallel"))
    assert _losses(h) == _losses(_run(_cfg()))


def test_parallel_session_keeps_client_states_apart_until_round_end():
    data = _data()
    calls = []

    def fedavg(states, info):
        calls.append(info["epoch"])
        avg = {i: {"model": {k: sum(s[i]["model"][k] for s in states) / len(states)
                             for k in states[0][i]["model"]},
                   "optimizer": states[0][i]["optimizer"]} for i in states[0]}
        return [avg for _ in states]

    apart = ts.ClientSession([data[:2], data[2:]], mode="parallel")
    _run(_cfg(setting="u_shaped"), data_fn=lambda: apart)
    w = lambda s: s[0]["model"]["layers.0.weight"]
    assert set(apart.states[0]) == {0, 2}                    # u_shaped: front and tail
    assert not torch.equal(w(apart.states[0]), w(apart.states[1]))
    avg = ts.ClientSession([data[:2], data[2:]], mode="parallel", on_round_end=fedavg)
    _run(_cfg(setting="u_shaped"), data_fn=lambda: avg)
    assert calls == [1, 2] and torch.equal(w(avg.states[0]), w(avg.states[1]))


def test_sequential_keep_handoff_shares_only_the_server():
    data = _data()
    s = ts.ClientSession([data[:2], data[2:]], mode="sequential", handoff="keep")
    h = _run(_cfg(), data_fn=lambda: s)
    assert _losses(h) != _losses(_run(_cfg()))
    w0, w1 = (st[0]["model"]["layers.0.weight"] for st in s.states)
    assert not torch.equal(w0, w1)


def test_sessions_need_client_slices():
    with pytest.raises(ValueError, match="needs client slices"):
        _run(_cfg(setting="pipeline"), data_fn=lambda: ts.ClientSession([_data()]))


def test_sessions_refuse_ties_across_client_and_server():
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(10, 8)
            self.mid = nn.Linear(8, 8)
            self.head = nn.Linear(8, 10, bias=False)
            self.head.weight = self.emb.weight

        def forward(self, x):
            return self.head(self.mid(self.emb(x)))

    data = [(torch.randint(0, 10, (4,)), torch.randint(0, 10, (4,)))]
    sliced = ts.slice(Tied(), cuts=["emb", "mid"])
    with pytest.raises(ValueError, match="tied across a client slice"):    # head on the server
        sliced.train(ts.ClientSession([data]), SGD, CE, epochs=1, run_config=_cfg(setting="vanilla"))
    # u_shaped: embedding and head are both on the data owner, so the tie is fine
    ts.slice(Tied(), cuts=["emb", "mid"]).train(ts.ClientSession([data, data], mode="parallel"),
                                                SGD, CE, epochs=1, run_config=_cfg(setting="u_shaped"))
    # frozen tied weight (LoRA-style fine-tuning): every client sees the same value, allowed
    frozen = Tied()
    frozen.emb.weight.requires_grad_(False)
    before = frozen.emb.weight.detach().clone()
    ts.slice(frozen, cuts=["mid"]).train(ts.ClientSession([data, data]), SGD, CE, epochs=1,
                                         run_config=_cfg(setting="vanilla"))
    assert torch.equal(frozen.emb.weight, before)


def test_run_wraps_client_loaders_from_the_config():
    data = _data()
    h = _run(_cfg(clients=2, client_mode="sequential"), data_fn=lambda: [data[:2], data[2:]])
    assert _losses(h) == _losses(_run(_cfg()))
    with pytest.raises(ValueError, match="must return 2 loaders"):
        _run(_cfg(clients=2), data_fn=lambda: [data])


# ── distributed: same training, same metrics ───────────────────────────────────

@pytest.mark.parametrize("topology,setting,transport", [
    ("centralized", "pipeline", "grpc"),
    ("centralized", "vanilla", "tcp"),
    ("centralized", "u_shaped", "grpc"),
    ("p2p", "vanilla", "grpc"),
    ("p2p", "u_shaped", "tcp"),
])
def test_distributed_settings_match_local(topology, setting, transport):
    local = _run(_cfg(setting=setting))
    got = _run(_cfg(topology, setting, transport))
    assert _losses(got) == pytest.approx(_reference(_data()), abs=1e-6)
    assert _losses(got) == pytest.approx(_losses(local), abs=1e-6)
    for k in METRICS:
        assert got[0][k] == local[0][k], k
    assert len(got[0]["compute_s"]) == 3 and got[0]["sl_setting"] == setting


@pytest.mark.parametrize("hook", [ts.Quantize(bits=4), ts.TopK(0.25, grad=True),
                                  ts.GaussianNoise(0.1, grad=True)])
def test_distributed_hooks_match_local(hook):
    local = _run(_cfg(), cut_hooks=hook)
    for topology, setting, transport in [("centralized", "pipeline", "tcp"), ("p2p", "u_shaped", "grpc")]:
        got = _run(_cfg(topology, setting, transport), cut_hooks=hook)
        assert _losses(got) == pytest.approx(_losses(local), abs=1e-6)
        assert got[0]["bytes_fwd"] == local[0]["bytes_fwd"]
        assert got[0]["bytes_bwd"] == local[0]["bytes_bwd"]


def test_distributed_gpipe_with_hooks_matches_local():
    cfg = _cfg()
    cfg.pipeline.use_gpipe, cfg.pipeline.n_micro = True, 2
    local = _run(cfg, cut_hooks={0: ts.Quantize(8), 1: ts.Float16()})
    cfg = _cfg("p2p", "u_shaped")
    cfg.pipeline.use_gpipe, cfg.pipeline.n_micro = True, 2
    got = _run(cfg, cut_hooks={0: ts.Quantize(8), 1: ts.Float16()})
    assert _losses(got) == pytest.approx(_losses(local), abs=1e-6)
    assert got[0]["bytes_fwd"] == local[0]["bytes_fwd"]


@pytest.mark.parametrize("mode", ["sequential", "parallel"])
@pytest.mark.parametrize("topology,setting", [("centralized", "vanilla"), ("p2p", "u_shaped")])
def test_distributed_sessions_match_local(mode, topology, setting):
    data = _data()
    two = lambda: [data[:2], data[2:]]
    local = _run(_cfg(setting=setting, clients=2, client_mode=mode), data_fn=two)
    got = _run(_cfg(topology, setting, clients=2, client_mode=mode), data_fn=two)
    assert _losses(got) == pytest.approx(_losses(local), abs=1e-6)


def test_raw_data_and_labels_stay_on_the_data_owner():
    """With vanilla, no worker sees a raw input; with u_shaped, no worker sees a label
    either: the data owner's slices run in-process."""
    seen = {"input_shapes": set(), "labels": 0}

    class Spy(ts.WorkerServicer):
        def forward(self, request, context):
            if request.label.data:
                seen["labels"] += 1
            if request.input.data:
                seen["input_shapes"].add(tuple(request.input.shape))
            return super().forward(request, context)

    for setting, labels in [("vanilla", 4 * 2), ("u_shaped", 0)]:
        seen.update(input_shapes=set(), labels=0)
        cfg = _cfg("centralized", setting)
        cfg.discovery.backend = "static"
        n_remote = 3 - (1 if setting == "vanilla" else 2)
        workers = [ts.start_worker(host="127.0.0.1", run_config=cfg, servicer_class=Spy)
                   for _ in range(n_remote)]
        try:
            port = ts.deploy.runner.free_port()
            ex = ts.DistributedExecutor(ts.StaticDiscovery([w.address for w in workers]),
                                        coordinator_addr=f"127.0.0.1:{port}",
                                        coordinator_bind_addr=f"127.0.0.1:{port}",
                                        run_config=cfg)
            if setting == "u_shaped":
                cfg.sl.tail_port = ts.deploy.runner.free_port()
            h = ts.slice(_model(), cuts=CUTS, executor=ex).train(_data(), run_config=cfg)
        finally:
            for w in workers:
                w.stop()
        assert _losses(h) == pytest.approx(_reference(_data()), abs=1e-6)
        assert (B, D_IN) not in seen["input_shapes"]              # raw inputs never sent
        assert seen["input_shapes"] <= {(B, D_H)}                 # only cut activations
        assert seen["labels"] == labels and h[0]["bytes_label"] == (4 * B * 8 if labels else 0)


class Scale(nn.Module):      # module level: slices are pickled to the workers
    def forward(self, x):
        return x.float() / 255


def test_coordinator_sends_uint8_inputs():
    """Regression: uint8 inputs were sent as float32 bytes."""
    def model():
        torch.manual_seed(0)
        return nn.Sequential(Scale(), nn.Linear(D_IN, D_H), nn.ReLU(), nn.Linear(D_H, 4))

    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 255, (32, D_IN), dtype=torch.uint8, generator=g)
    y = torch.randint(0, 4, (32,), generator=g)
    data = lambda: [(x[i:i + 8], y[i:i + 8]) for i in range(0, 32, 8)]
    cfg = _cfg("centralized", "pipeline")
    cfg.discovery.n_workers = 2
    got = ts.run(model, data, run_config=cfg, verbose=False, cuts=["2"])
    local = _cfg()
    local.discovery.n_workers = 2
    ref = ts.run(model, data, run_config=local, verbose=False, cuts=["2"])
    assert _losses(got) == pytest.approx(_losses(ref), abs=1e-6)
    assert got[0]["bytes_input"] == 32 * D_IN                    # 1 byte per value


def test_late_labels_are_not_kept_on_the_last_worker(monkeypatch):
    """Regression: a label arriving after its activation was
    used and also stored, so the last worker kept every such label until the run ended."""
    from torchslicer.executors import distributed as dist
    left = []

    class Last(ts.WorkerServicer):
        def shutdown(self, request, context):
            if self.is_last:
                left.append(len(self._labels))
            return super().shutdown(request, context)

    class Late:
        def __init__(self, stub):
            self._stub = stub

        def __getattr__(self, name):
            return getattr(self._stub, name)

        def forward(self, request, **kw):
            if request.label.data:   # deliver the label after the activation
                threading.Timer(0.2, lambda: self._stub.forward(request)).start()
                return None
            return self._stub.forward(request, **kw)

    stub = dist._WorkerProxy.stub
    monkeypatch.setattr(dist._WorkerProxy, "stub",
                        lambda self: Late(stub(self)) if self.worker_index == 1 else stub(self))
    cfg = _cfg("centralized", "pipeline")
    cfg.discovery.n_workers = 2
    workers = [ts.start_worker(host="127.0.0.1", run_config=cfg, servicer_class=Last) for _ in range(2)]
    try:
        port = ts.deploy.runner.free_port()
        ex = ts.DistributedExecutor(ts.StaticDiscovery([w.address for w in workers]),
                                    coordinator_addr=f"127.0.0.1:{port}",
                                    coordinator_bind_addr=f"127.0.0.1:{port}", run_config=cfg)
        h = ts.slice(_model(), cuts=["3"], executor=ex).train(_data(), run_config=cfg)
    finally:
        time.sleep(0.3)
        for w in workers:
            w.stop()
    assert _losses(h) == pytest.approx(_reference(_data()), abs=1e-6)
    assert left == [0]


# ── deployment: node counts, tail port ─────────────────────────────────────────

@pytest.mark.parametrize("topology,setting,roles", [
    ("centralized", "", ["coordinator", "worker", "worker", "worker"]),
    ("centralized", "vanilla", ["coordinator", "worker", "worker"]),
    ("centralized", "u_shaped", ["coordinator", "worker"]),
    ("p2p", "", ["driver", "follower", "follower"]),
    ("p2p", "u_shaped", ["driver", "follower"]),
])
def test_node_plan_follows_the_setting(topology, setting, roles):
    cfg = _cfg(topology, setting)
    assert [r for _, r in node_names(cfg)] == roles
    nodes = plan_nodes(cfg, ["h"] * len(roles), [5000 + 10 * i for i in range(len(roles))])
    assert ("SL_TAIL_PORT" in nodes[0].env) == (setting == "u_shaped")
    if setting == "u_shaped":
        assert nodes[0].env["SL_TAIL_PORT"] == "5002"


def test_pipeline_is_refused_in_p2p():
    with pytest.raises(ValueError, match="not possible in P2P"):
        node_names(_cfg("p2p", "pipeline"))


def test_k8s_exposes_the_tail_port():
    cfg = _cfg("p2p", "u_shaped", transport="tcp")
    docs = render_k8s(cfg, "train.py", "run_id: x\n")
    svc = next(d for d in docs if d["kind"] == "Service" and "driver" in d["metadata"]["name"])
    names = {p["name"]: p["port"] for p in svc["spec"]["ports"]}
    port = cfg.network.worker_port
    assert names == {"grpc": port, "tensor": port + 1, "tail": port + 2, "tail-tensor": port + 3}


def test_baremetal_tail_port_does_not_collide(tmp_path):
    cfg = _cfg("p2p", "u_shaped", transport="tcp")
    cfg.discovery.n_workers = 4                     # driver + 2 followers, all on one host
    paths = render_baremetal(cfg, "train.py", "c.yaml", ["h"] * 3, out=str(tmp_path), workdir=".")
    env = {}
    for p in paths:
        for line in open(p):
            if line.startswith("export "):
                k, v = line[7:].strip().split("=", 1)
                env.setdefault(os.path.basename(p), {})[k] = v
    used = set()
    for e in env.values():
        port = int(e.get("PORT", 0))
        used |= {port, port + 1}
    tail = int(env["driver.sh"]["SL_TAIL_PORT"])
    assert not {tail, tail + 1} & used


def _launch(*args, env=None):
    r = subprocess.run([sys.executable, "-m", "torchslicer", "launch",
                        "examples/run_anywhere/train.py", *args],
                       cwd=ROOT, capture_output=True, text=True, timeout=240,
                       env={**os.environ, **(env or {})})
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    line = next(l for l in r.stdout.splitlines() if l.startswith("HISTORY "))
    return json.loads(line[len("HISTORY "):])


@pytest.mark.parametrize("config,env", [
    ("p2p", {"SL_SETTING": "u_shaped", "TENSOR_TRANSPORT": "tcp"}),
    ("centralized", {"SL_SETTING": "vanilla"}),
])
def test_launch_processes_with_sl_settings_matches_local(config, env, tmp_path):
    local = _launch("-c", f"experiments/run_anywhere_{config}.yaml", "--target", "local")
    got = _launch("-c", f"experiments/run_anywhere_{config}.yaml", "--target", "processes",
                  "-o", str(tmp_path), env=env)
    assert got == pytest.approx(local, abs=1e-6)


# ── fault tolerance with the data owner's slices ───────────────────────────────

class _StopAfter(ts.TrainingCallback):
    def __init__(self, worker, epoch):
        self.worker, self.epoch = worker, epoch

    def on_epoch_end(self, epoch, metrics):
        if epoch == self.epoch:
            self.worker.stop()
        return metrics


def test_u_shaped_recovery_keeps_the_tail_on_the_data_owner(tmp_path):
    """4 slices: the driver hosts 0 and 3. A follower dies; the survivors re-plan to 3
    slices, and the driver still hosts the first and the last."""
    def model():
        torch.manual_seed(0)
        return nn.Sequential(*[m for _ in range(3) for m in (nn.Linear(D_H, D_H), nn.ReLU())],
                             nn.Linear(D_H, 4))

    def data():
        g = torch.Generator().manual_seed(1)
        x, y = torch.randn(64, D_H, generator=g), torch.randint(0, 4, (64,), generator=g)
        return [(x[i:i + B], y[i:i + B]) for i in range(0, 64, B)]

    cfg = _cfg(setting="u_shaped")
    cfg.training.epochs = 3
    cfg.fault_tolerance.enabled, cfg.fault_tolerance.max_retries = True, 1
    cfg.fault_tolerance.heartbeat_interval_s, cfg.fault_tolerance.ping_timeout_s = 0.2, 0.5
    cfg.checkpoint.dir = str(tmp_path)
    cfg.sl.tail_port = ts.deploy.runner.free_port()
    followers = [ts.start_worker(host="127.0.0.1", run_config=cfg) for _ in range(2)]
    try:
        port = ts.deploy.runner.free_port()
        addr = f"127.0.0.1:{port}"
        ex = ts.P2PExecutor(peers=[w.address for w in followers], address=addr, bind_addr=addr,
                            run_config=cfg)
        h = ts.slice(model(), n=4, executor=ex).train(
            data(), run_config=cfg, callbacks=[_StopAfter(followers[1], 1)])
    finally:
        for w in followers:
            w.stop()
    assert len(h) == 3 and all(e["loss"] == e["loss"] for e in h)
    assert [nd.address for nd in ex._nodes] == [addr, followers[0].address, f"127.0.0.1:{cfg.sl.tail_port}"]
    assert sorted(ex._local) == [0, 2] and len(ex.partitions) == 3
    assert h[-1]["bytes_label"] == 0 and len(h[-1]["bytes_fwd"]) == 2


def test_distributed_refuses_algorithm_hooks_it_does_not_run():
    # It used to ignore them and train plain SplitNN instead.
    data = _data()
    two = lambda: [data[:2], data[2:]]
    cfg = _cfg("p2p", "vanilla", clients=2, client_mode="parallel", algorithm="splitfed_v1")
    with pytest.raises(ValueError, match="per-client server copies"):
        _run(cfg, data_fn=two)


# ── client-state store (M11 step 3) ──────────────────────────────────────────

def _session_run(cfg, mode="sequential", handoff="copy", opt=None, spy=None):
    data = _data()
    s = ts.ClientSession([data[:2], data[2:]], mode=mode, handoff=handoff)
    if spy:
        spy(s)
    if opt:
        cfg.training.optimizer = opt
    return s, ts.slice(_model(), n=3, cuts=CUTS).train(s, run_config=cfg)


def test_stored_states_are_never_aliased_by_the_live_slices():
    # Adam's state (and its in-place step counter) must not leak back into a stored
    # state: with handoff "copy", client 1 starts from client 0's stored state.
    snaps = {}
    def spy(s):
        orig = s._switch
        def switch(adapter, k, epoch):
            orig(adapter, k, epoch)
            if k == 1 and epoch == 1:
                import copy as _c
                snaps["c0"] = _c.deepcopy(s.states[0])
        s._switch = switch
    cfg = _cfg(setting="vanilla")
    cfg.training.epochs = 1
    s, _ = _session_run(cfg, opt={"name": "Adam", "params": {"lr": 0.01}}, spy=spy)
    after = s.states[0][0]
    assert all(torch.equal(after["model"][k], v) for k, v in snaps["c0"][0]["model"].items())
    for pid, st in snaps["c0"][0]["optimizer"]["state"].items():
        for key, v in st.items():
            assert torch.equal(after["optimizer"]["state"][pid][key], v), key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("mode", ["sequential", "parallel"])
def test_gpu_client_store_matches_cpu_store(mode):
    def run(store):
        cfg = _cfg(setting="vanilla", client_states=store)
        cfg.network.device = "cuda"
        torch.backends.cudnn.deterministic = True
        s = ts.ClientSession([_data()[:2], _data()[2:]], mode=mode)
        h = ts.slice(_model().cuda(), n=3, cuts=CUTS).train(s, run_config=cfg)
        return s, h
    (sc, hc), (sg, hg) = run("cpu"), run("gpu")
    assert _losses(hg) == pytest.approx(_losses(hc), abs=1e-6)
    assert sg.states[0][0]["model"]["layers.0.weight"].is_cuda
    assert not sc.states[0][0]["model"]["layers.0.weight"].is_cuda


def test_client_states_that_do_not_fit_stop_before_training(monkeypatch):
    import torchslicer.sl.clients as clients
    monkeypatch.setattr(clients, "available_bytes", lambda device: 1000)
    with pytest.raises(ValueError, match="client states need"):
        _session_run(_cfg(setting="vanilla"))
    # auto falls back to the host when the GPU is too small
    monkeypatch.setattr(clients, "available_bytes",
                        lambda device: 1000 if str(device).startswith("cuda") else 1 << 40)
    layers = {0: ts.SplitLayer([nn.Linear(4, 4)])}
    assert clients.resolve_client_store("auto", layers, 10) == "cpu"


def test_state_nbytes_counts_optimizer_slots():
    from torchslicer.sl.clients import state_nbytes
    def layer(opt):
        sl = ts.SplitLayer([nn.Linear(10, 10)])           # 110 params, 440 bytes
        sl.set_optimizer(opt(sl.parameters()))
        return {0: sl}
    assert state_nbytes(layer(lambda p: torch.optim.SGD(p, lr=0.1))) == 440
    assert state_nbytes(layer(lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9))) == 880
    assert state_nbytes(layer(lambda p: torch.optim.Adam(p))) == 1320


def test_distributed_refuses_simulated_client_profiles():
    data = _data()
    cfg = _cfg("p2p", "vanilla", clients=2, client_mode="parallel",
               client_profiles=[{"compute_scale": 2.0}])
    with pytest.raises(ValueError, match="client_profiles"):
        _run(cfg, data_fn=lambda: [data[:2], data[2:]])
