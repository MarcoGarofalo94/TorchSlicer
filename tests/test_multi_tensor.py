"""M18 multi-tensor cuts: several values cross a cut, packed into one tensor by a graph rewrite.

The invariant: a model cut through its skips (``ts.slice(..., multi_tensor=True)``) trains
exactly like the unsplit one, in every executor and topology.
"""
import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchslicer as ts
from torchslicer.config import RunConfig
from torchslicer.core.multicut import _RecvCut, _SendCut, crossing, pack_cuts, packed_cuts

SGD = {"name": "SGD", "params": {"lr": 0.05, "momentum": 0.9}}
CE = {"name": "CrossEntropyLoss"}


# ── models whose skips leave few single-tensor cuts ─────────────────────────────

class UNet(nn.Module):
    def __init__(self, c=4, bn=True):
        super().__init__()
        norm = (lambda k: nn.BatchNorm2d(k)) if bn else (lambda k: nn.Identity())
        self.e1 = nn.Sequential(nn.Conv2d(3, c, 3, padding=1), norm(c), nn.ReLU())
        self.e2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(c, 2 * c, 3, padding=1), nn.ReLU())
        self.mid = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(2 * c, 2 * c, 2, stride=2)
        self.d2 = nn.Sequential(nn.Conv2d(4 * c, c, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(c, c, 2, stride=2)
        self.d1 = nn.Conv2d(2 * c, 3, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(a)
        h = self.d2(torch.cat([self.up2(self.mid(b)), b], 1))
        return self.d1(torch.cat([self.up1(h), a], 1))


class InputSkip(nn.Module):
    """Reads the model input again at the end; a parameter crosses the cuts too."""

    def __init__(self):
        super().__init__()
        self.f, self.g, self.h = nn.Linear(6, 6), nn.Linear(6, 6), nn.Linear(6, 3)
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        return self.h(self.g(self.f(x)) * self.scale + x)


class DecLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, 2, batch_first=True)
        self.ff = nn.Linear(d, d)

    def forward(self, h, mem):
        return h + self.ff(self.attn(h, mem, mem)[0])


class EncDec(nn.Module):
    """Every decoder layer cross-attends to the encoder output; the decoder queries are a
    parameter expanded to the batch size (a Python int crosses the cuts)."""

    def __init__(self, d=8, n_dec=3):
        super().__init__()
        self.emb = nn.Linear(4, d)
        self.enc = nn.TransformerEncoderLayer(d, 2, 16, dropout=0.0, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 5, d))
        self.dec = nn.ModuleList(DecLayer(d) for _ in range(n_dec))
        self.head = nn.Linear(d, 3)

    def forward(self, x):
        mem = self.enc(self.emb(x))
        h = self.query.expand(x.size(0), -1, -1)
        for layer in self.dec:
            h = layer(h, mem)
        return self.head(h.mean(1))


def _case(name):
    g = torch.Generator().manual_seed(7)
    if name == "unet":
        x, y = torch.randn(16, 3, 8, 8, generator=g), torch.randint(0, 3, (16, 8, 8), generator=g)
        return lambda bn=True: UNet(bn=bn), x, y
    if name == "input_skip":
        return lambda bn=True: InputSkip(), torch.randn(16, 6, generator=g), torch.randint(0, 3, (16,), generator=g)
    return lambda bn=True: EncDec(), torch.randn(16, 7, 4, generator=g), torch.randint(0, 3, (16,), generator=g)


MODELS = ["unet", "input_skip", "enc_dec"]


def _fresh(make, bn=True):
    torch.manual_seed(0)
    return make(bn).train()


def _batches(x, y, size=4):
    return [(x[i:i + size], y[i:i + size]) for i in range(0, len(x), size)]


def _unsplit(model, data, epochs=1):
    opt = torch.optim.SGD(model.parameters(), **SGD["params"])
    for _ in range(epochs):
        for x, y in data:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    return model


def _same(a, b, atol=1e-6):
    for (name, x), y in zip(a.state_dict().items(), b.state_dict().values()):
        assert torch.allclose(x, y, atol=atol), name


# ── the graph rewrite ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", MODELS)
def test_skips_leave_few_single_tensor_cuts_and_multi_tensor_allows_every_gap(name):
    make, _, _ = _case(name)
    g = ts.trace(_fresh(make))
    assert len(g.valid_cuts()) < len(g) - 1
    gaps = [k for k in range(len(g) - 1) if k not in g.valid_cuts()]
    assert all(len(crossing(g, k)) > 1 or crossing(g, k) != [k] for k in gaps)
    s = ts.slice(g, cuts=[g.names()[gaps[0]]], multi_tensor=True)
    assert [type(n.module) for n in s.graph.nodes].count(_SendCut) == 1
    send = next(n for n in s.graph.nodes if isinstance(n.module, _SendCut))
    assert s.partitions[0].layer_indices[-1] == send.index
    recvs = [n for n in s.graph.nodes if isinstance(n.module, _RecvCut)]
    assert len(recvs) == len(crossing(g, gaps[0])) and all(n.predecessors == [send.index] for n in recvs)


def test_valid_cuts_are_left_alone():
    make, _, _ = _case("unet")
    g = ts.trace(_fresh(make))
    packed, cuts = pack_cuts(g, g.valid_cuts())
    assert packed is g and cuts == g.valid_cuts()
    s = ts.slice(_fresh(make), cuts=[g.names()[g.valid_cuts()[0]]], multi_tensor=True)
    assert s.graph.names() == g.names()


def test_without_multi_tensor_a_skip_cut_is_still_refused():
    make, _, _ = _case("unet")
    with pytest.raises(ValueError):
        ts.slice(_fresh(make), cuts=["e2"])


# ── the packing ────────────────────────────────────────────────────────────────

def test_pack_round_trips_every_kind_of_value_and_its_gradient():
    torch.manual_seed(0)
    a = torch.randn(3, 2, 4, requires_grad=True)                     # per row, float32
    b = torch.randn(3, 5, dtype=torch.float16, requires_grad=True)   # promoted, cast back
    p = torch.randn(4, 4, requires_grad=True)                        # repeated (no batch dim)
    ids = torch.randint(0, 50_000, (3, 7))                           # int64, as bytes
    mask = torch.rand(3, 7) > 0.5                                    # bool, as bytes
    values = [a, b, p, ids, mask, 3, 0.25, True]
    packed = _SendCut()(*values)
    assert packed.shape[0] == 3 and packed.dtype == torch.float32
    got = [_RecvCut(i)(packed) for i in range(len(values))]
    for v, w in zip(values, got):
        if isinstance(v, torch.Tensor):
            assert w.dtype == v.dtype and torch.equal(w, v)
        else:
            assert type(w) is type(v) and w == v
    loss = (got[0] ** 2).sum() + (got[1].float() * 3).sum() + (got[2] @ got[2]).sum()
    loss.backward()
    ref = [t.detach().clone().requires_grad_(True) for t in (a, b, p)]
    ((ref[0] ** 2).sum() + (ref[1].float() * 3).sum() + (ref[2] @ ref[2]).sum()).backward()
    for t, r in zip((a, b, p), ref):
        assert torch.allclose(t.grad, r.grad)


def test_bytes_per_cut_are_the_packed_tensor():
    make, x, y = _case("input_skip")
    s = ts.slice(_fresh(make), cuts=["f"], multi_tensor=True)
    hist = s.train(_batches(x, y), SGD, CE, epochs=1)
    with torch.no_grad():
        packed = s.slices()[0].run(x[:4])
    assert hist[0]["bytes_fwd"][0] == len(_batches(x, y)) * packed.numel() * packed.element_size()


# ── exactness in every executor ────────────────────────────────────────────────

@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("n", [2, 3, 4])
def test_local_trains_like_unsplit(name, n):
    make, x, y = _case(name)
    ref = _unsplit(_fresh(make), _batches(x, y))
    model = _fresh(make)
    ts.slice(model, n=n, multi_tensor=True).train(_batches(x, y), SGD, CE, epochs=1)
    _same(model, ref)


@pytest.mark.parametrize("name", MODELS)
def test_gpipe_trains_like_unsplit(name):
    make, x, y = _case(name)
    data = _batches(x, y, size=8)
    ref = _unsplit(_fresh(make, bn=False), data)
    model = _fresh(make, bn=False)
    ts.slice(model, n=3, multi_tensor=True).train(data, SGD, CE, epochs=1, n_micro_batches=2)
    _same(model, ref, atol=1e-5)


@pytest.mark.parametrize("mode", ["cpu", "disk"])
def test_offload_trains_like_unsplit(mode, tmp_path):
    make, x, y = _case("enc_dec")
    ref = _unsplit(_fresh(make), _batches(x, y))
    model = _fresh(make)
    executor = ts.OffloadExecutor(mode=mode, device="cpu", directory=str(tmp_path))
    ts.slice(model, n=3, multi_tensor=True, executor=executor).train(_batches(x, y), SGD, CE, epochs=1)
    _same(model, ref, atol=1e-5)


def _cfg(topology, setting, transport="grpc"):
    c = RunConfig()
    c.training.epochs = 1
    c.training.optimizer, c.training.criterion = SGD, CE
    c.discovery.n_workers = 3
    c.deploy.topology = topology
    c.network.device = "cpu"
    c.logging.enabled = False
    c.transport.tensor = transport
    c.sl.setting = setting
    return c


@pytest.mark.parametrize("topology,setting,transport", [
    ("centralized", "pipeline", "grpc"),
    ("centralized", "u_shaped", "tcp"),
    ("p2p", "vanilla", "tcp"),
])
def test_cluster_trains_like_unsplit(topology, setting, transport):
    make, x, y = _case("unet")
    ref = _unsplit(_fresh(make), _batches(x, y))
    model = _fresh(make)
    ts.run(lambda: model, lambda: _batches(x, y), run_config=_cfg(topology, setting, transport),
           verbose=False, multi_tensor=True)
    _same(model, ref, atol=1e-5)


def test_the_config_turns_multi_tensor_on_for_ts_run(tmp_path, monkeypatch):
    make, x, y = _case("unet")
    ref = _unsplit(_fresh(make), _batches(x, y))
    (tmp_path / "c.yaml").write_text("planning:\n  multi_tensor: true\n")
    assert RunConfig.load(str(tmp_path / "c.yaml")).planning.multi_tensor
    monkeypatch.setenv("PLANNING_MULTI_TENSOR", "false")
    assert not RunConfig.load(str(tmp_path / "c.yaml")).planning.multi_tensor
    cfg = _cfg("centralized", "pipeline")
    cfg.planning.multi_tensor = True
    model = _fresh(make)
    ts.run(lambda: model, lambda: _batches(x, y), run_config=cfg, verbose=False, cuts=["e2", "d2"])
    _same(model, ref, atol=1e-5)


# ── sessions, evaluation, tied weights ─────────────────────────────────────────

def _cut_weights(cuts, mode, algorithm=None):
    make, x, y = _case("unet")
    data = _batches(x, y)
    model = _fresh(make)
    s = ts.slice(model, cuts=cuts, multi_tensor=True)
    s.train(ts.ClientSession([data[:2], data[2:]], mode=mode), SGD, CE, epochs=1,
            algorithm=algorithm() if algorithm else None)
    return model


@pytest.mark.parametrize("mode,algorithm", [("sequential", None), ("parallel", ts.SplitFedAlgorithm)])
def test_client_session_trains_like_single_tensor_cuts(mode, algorithm):
    """Same client slice (e1); the server part cut through the U-Net's skips, or not."""
    g = ts.trace(_fresh(_case("unet")[0]))
    _same(_cut_weights(["e1", "mid"], mode, algorithm), _cut_weights(["e1", g.names()[g.valid_cuts()[-1]]], mode, algorithm))


def test_evaluation_runs_through_packed_cuts():
    make, x, y = _case("unet")                     # no BatchNorm: training at lr 0 changes nothing
    model = _fresh(make, bn=False).eval()
    with torch.no_grad():
        want = F.cross_entropy(model(x[:4]), y[:4]).item()
    hist = ts.slice(_fresh(make, bn=False), n=3, multi_tensor=True).train(
        _batches(x, y)[:1], {"name": "SGD", "params": {"lr": 0.0}}, CE, epochs=1, eval_data=_batches(x, y)[:1])
    assert hist[0]["test_loss"] == pytest.approx(want, abs=1e-6)


class TiedSkip(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.f = nn.Embedding(20, 8), nn.Linear(8, 8)
        self.head = nn.Linear(8, 20, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, t):
        e = self.emb(t)
        return self.head(self.f(e) + e).transpose(1, 2)


def test_tied_weights_across_packed_cuts_train_like_unsplit():
    torch.manual_seed(0)
    base, t, y = TiedSkip(), torch.randint(0, 20, (4, 6)), torch.randint(0, 20, (4, 6))
    ref = copy.deepcopy(base)
    opt = torch.optim.Adam(ref.parameters(), lr=0.01)
    for _ in range(3):
        opt.zero_grad()
        F.cross_entropy(ref(t), y).backward()
        opt.step()
    model = copy.deepcopy(base)
    s = ts.slice(model, cuts=["emb", "f"], multi_tensor=True)
    assert len(s.partitions) == 3
    s.train([(t, y)] * 3, {"name": "Adam", "params": {"lr": 0.01}}, CE, epochs=1)
    _same(model, ref)
    assert model.head.weight is model.emb.weight


# ── real architectures (torchvision): every cut through their skips is exact ───

def _tv(name):
    tvm = pytest.importorskip("torchvision.models")
    return {"resnet18": lambda: tvm.resnet18(num_classes=10),
            "regnet": lambda: tvm.regnet_y_400mf(num_classes=10),     # in-place ReLU after cuts
            "swin": lambda: tvm.swin_t(num_classes=10)}[name]         # weights read in forward


@pytest.mark.parametrize("name", ["resnet18", "regnet", "swin"])
def test_torchvision_models_cut_through_their_skips_train_like_unsplit(name):
    torch.manual_seed(0)
    base = _tv(name)().train()
    g = torch.Generator().manual_seed(2)
    x, y = torch.randn(4, 3, 64, 64, generator=g), torch.randint(0, 10, (4,), generator=g)
    ref = copy.deepcopy(base)
    torch.manual_seed(1)
    _unsplit(ref, [(x, y)])
    model = copy.deepcopy(base)
    s = ts.slice(model, n=4, expand="all", multi_tensor=True)
    assert packed_cuts(s.graph, s.partitions)
    torch.manual_seed(1)
    h = s.train([(x, y)], SGD, CE, epochs=1)
    _same(model, ref, atol=1e-6)
    # no weight crosses a cut, and the batch is the data's: a cut is a few activations
    assert max(h[0]["bytes_fwd"]) < 2 * 2**20


def test_a_weight_or_a_table_built_from_weights_never_crosses_a_cut():
    class Biased(nn.Module):
        def __init__(self):
            super().__init__()
            self.w, self.table = nn.Parameter(torch.randn(6, 6)), nn.Parameter(torch.randn(6))
            self.out = nn.Linear(6, 3)

        def forward(self, x):
            bias = self.table * 2                    # built from a weight only
            h = torch.relu(x @ self.w) + bias
            return self.out(h + x)

    g = ts.trace(Biased())
    anywhere = g.multi_tensor_cuts()
    # input | table, mul | w | matmul, relu | add ...: mul (from a weight) is read by add, w by
    # matmul, so only the gaps after input, add and add_1 are left (x crosses those as data)
    assert [g.nodes[k].name for k in anywhere] == ["input", "add", "add_1"]
    s = ts.slice(Biased(), n=3, multi_tensor=True)           # uniform counts nodes: snapped
    assert set(s.graph.base_cuts) <= set(anywhere)
    with pytest.raises(ValueError, match="a weight"):
        ts.slice(Biased(), cuts=["mul"], multi_tensor=True)


def test_a_slice_that_only_builds_the_mask_needs_no_backward():
    torch.manual_seed(0)
    ids, y = torch.randint(0, 50, (4, 12)), torch.randint(0, 50, (4, 12))
    ref = TinyDecoder()
    model = copy.deepcopy(ref)
    opt = torch.optim.SGD(ref.parameters(), **SGD["params"])
    F.cross_entropy(ref(ids), y).backward()
    opt.step()
    s = ts.slice(model, cuts=["triu", "add_2"], multi_tensor=True)
    assert not list(s.slices()[0].parameters())              # slice 0: positions and the mask
    s.train([(ids, y)], SGD, CE, epochs=1)
    _same(model, ref)


# ── cut hooks on a packed cut: the hook sees the values, not the layout ──────────

class _Seen(ts.CutHook):
    """Identity that records what it is given."""

    def __init__(self):
        self.seen = []

    def encode(self, x, ctx):
        self.seen.append((ctx.direction, tuple(x.shape), x.dtype))
        return x

    def encode_grad(self, g, ctx):
        self.seen.append((ctx.direction, tuple(g.shape), g.dtype))
        return g


@pytest.mark.parametrize("name", MODELS)
def test_an_identity_hook_on_packed_cuts_leaves_training_exact(name):
    make, x, y = _case(name)
    ref = _unsplit(_fresh(make), _batches(x, y))
    model = _fresh(make)
    hook = _Seen()
    s = ts.slice(model, n=3, multi_tensor=True, cut_hooks=hook)
    assert packed_cuts(s.graph, s.partitions)
    s.train(_batches(x, y), SGD, CE, epochs=1)
    _same(model, ref)
    assert {d for d, _, _ in hook.seen} == {"forward", "backward"}


def test_the_hook_sees_the_one_value_in_its_shape_or_all_values_flattened():
    hook = _Seen()
    torch.manual_seed(0)
    ids, y = torch.randint(0, 50, (4, 12)), torch.randint(0, 50, (4, 12))
    s = ts.slice(TinyDecoder(), cuts=["add_2"], multi_tensor=True, cut_hooks={0: hook})
    s.train([(ids, y)], SGD, CE, epochs=1)
    # the hidden state; the mask is rebuilt and the length crosses as bytes, unhooked
    assert hook.seen == [("forward", (4, 12, 16), torch.float32), ("backward", (4, 12, 16), torch.float32)]
    make, x, y = _case("unet")
    hook = _Seen()
    s = ts.slice(_fresh(make), cuts=["e2"], multi_tensor=True, cut_hooks={0: hook})
    s.train(_batches(x, y)[:1], SGD, CE, epochs=1)
    # e2 [4, 8, 4, 4] and e1 [4, 4, 8, 8] as one [4, 128 + 256] tensor per direction
    assert hook.seen == [("forward", (4, 384), torch.float32), ("backward", (4, 384), torch.float32)]


def test_a_per_sample_hook_bounds_everything_one_sample_sends():
    make, x, y = _case("unet")
    hook = ts.Recorder(what=("forward",))
    s = ts.slice(_fresh(make), cuts=["e2"], multi_tensor=True,
                 cut_hooks={0: ts.Chain([ts.ClipNoise(clip=0.5, noise_multiplier=0.0), hook])})
    s.train(_batches(x, y)[:1], SGD, CE, epochs=1)
    z = hook.records[0][2]
    assert z.shape == (4, 384) and torch.all(z.norm(dim=1) <= 0.5 + 1e-5)


def _hooked(run, hook_fn, **kw):
    make, x, y = _case("unet")
    model = _fresh(make, bn=False)
    run(model, _batches(x, y), hook_fn, **kw)
    return model


def _local_run(model, data, hook_fn, n_micro=1):
    ts.slice(model, cuts=["e2", "d2"], multi_tensor=True, cut_hooks=hook_fn()).train(
        data, SGD, CE, epochs=1, n_micro_batches=n_micro)


@pytest.mark.parametrize("hook_fn", [lambda: ts.Quantize(bits=8), lambda: ts.CircularConv(ratio=2, seed=3),
                                     lambda: ts.Chain([ts.Float16(), ts.TopK(ratio=0.5, grad=True)])])
def test_a_lossy_hook_on_packed_cuts_trains_the_same_everywhere(hook_fn):
    ref = _hooked(_local_run, hook_fn)
    plain = _hooked(_local_run, lambda: None)
    assert any(not torch.allclose(a, b) for a, b in zip(ref.parameters(), plain.parameters()))
    _same(_hooked(_local_run, hook_fn, n_micro=2), ref, atol=1e-5)   # GPipe: micro-batches of 2

    def cluster(model, data, hook_fn, topology, setting, transport):
        ts.run(lambda: model, lambda: data, run_config=_cfg(topology, setting, transport), verbose=False,
               cuts=["e2", "d2"], multi_tensor=True, cut_hooks=hook_fn())
    for topo in [("centralized", "pipeline", "grpc"), ("p2p", "vanilla", "tcp")]:
        _same(_hooked(cluster, hook_fn, topology=topo[0], setting=topo[1], transport=topo[2]), ref, atol=1e-5)


def test_hook_bytes_at_a_packed_cut_are_what_the_hook_sends():
    make, x, y = _case("unet")
    plain = ts.slice(_fresh(make), cuts=["e2"], multi_tensor=True).train(_batches(x, y), SGD, CE, epochs=1)
    half = ts.slice(_fresh(make), cuts=["e2"], multi_tensor=True, cut_hooks=ts.Float16()).train(
        _batches(x, y), SGD, CE, epochs=1)
    assert 0.45 < half[0]["bytes_fwd"][0] / plain[0]["bytes_fwd"][0] < 0.6


def test_attacks_are_refused_at_a_packed_cut():
    make, x, y = _case("input_skip")
    s = ts.slice(_fresh(make), cuts=["f"], multi_tensor=True)
    with pytest.raises(ValueError, match="packed"):
        s.train(_batches(x, y), SGD, CE, epochs=1, privacy=[ts.privacy.NormLabelAttack(cut=0)])


# ── recovery: survivors re-split the rewritten graph at its packed cuts ────────

@pytest.mark.parametrize("cuts", [None, ["e2", "d2"]])
def test_p2p_recovers_from_a_dead_follower_through_packed_cuts(cuts, tmp_path):
    from test_p2p import _StopAtStart, _cfg as p2p_cfg, _driver, _followers
    make, x, y = _case("unet")
    data = _batches(x, y)
    ref = _unsplit(_fresh(make), data, epochs=3)
    cfg = p2p_cfg(epochs=3, enabled=True, max_retries=1)
    cfg.checkpoint.dir = str(tmp_path)
    followers = _followers(cfg, 2)
    try:
        model = _fresh(make)
        s = ts.slice(model, n=None if cuts else 3, cuts=cuts, multi_tensor=True, executor=_driver(cfg, followers))
        assert sum(isinstance(n.module, _SendCut) for n in s.graph.nodes) == 2
        s.train(data, run_config=cfg, callbacks=[_StopAtStart(followers[1], 2)])
        assert len(s.partitions) == 2
        _same(model, ref, atol=1e-5)
        # the checkpoints rebuild the model: the graph is rewritten at the recorded cuts again
        _same(ts.restore_model_from_run(_fresh(make), str(tmp_path / cfg.run_id)), ref, atol=1e-5)
    finally:
        for w in followers:
            w.stop()


# ── client nodes: each client its own node, like the in-process session ────────

@pytest.mark.parametrize("cuts", [["e1", "mid"], ["e2", "d2"]])     # client cut single / packed
def test_client_nodes_train_like_the_in_process_session(cuts):
    from torchslicer.deploy.runner import free_port
    from torchslicer.discovery import StaticDiscovery
    make, x, y = _case("unet")
    data = _batches(x, y)
    clients_data = [data[:2], data[2:]]
    cfg = _cfg("local", "vanilla")
    cfg.sl.clients = 2
    cfg.discovery.timeout = 30
    ref = _fresh(make)
    ts.slice(ref, cuts=cuts, multi_tensor=True).train(ts.ClientSession(clients_data), run_config=cfg)
    offsets = cfg.transport.listen_offsets()
    servers = [ts.start_worker(free_port(offsets), host="127.0.0.1", run_config=cfg) for _ in range(2)]
    clients = [ts.start_worker(free_port(offsets), host="127.0.0.1", run_config=cfg, data=d, client_index=k)
               for k, d in enumerate(clients_data)]
    addr = f"127.0.0.1:{free_port(cfg.transport.listen_offsets(drives=True))}"
    ex = ts.ClientNodesExecutor(discovery=StaticDiscovery(peers=[w.address for w in clients + servers]),
                                coordinator_addr=addr, run_config=cfg)
    got = _fresh(make)
    try:
        ts.slice(got, cuts=cuts, multi_tensor=True, executor=ex).train(ts.RemoteSession(2), run_config=cfg)
    finally:
        for w in servers + clients:
            w.stop()
    _same(got, ref, atol=1e-5)


# ── recompute: values built from metadata and constants are rebuilt, not sent ──

class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, 2, batch_first=True)
        self.ff, self.n = nn.Linear(d, d), nn.LayerNorm(d)

    def forward(self, h, mask):
        return h + self.ff(self.n(h + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]))


class TinyDecoder(nn.Module):
    """Positions and a causal mask built in forward from the input's length and device; every
    block reads the mask (like an LLM: what crosses a block boundary is h + mask)."""

    def __init__(self, vocab=50, d=16, n=3, noise=False):
        super().__init__()
        self.emb, self.pos = nn.Embedding(vocab, d), nn.Embedding(64, d)
        self.blocks = nn.ModuleList(_Block(d) for _ in range(n))
        self.head = nn.Linear(d, vocab)
        self.noise = noise

    def forward(self, ids):
        T = ids.size(1)
        pos = torch.arange(T, device=ids.device)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), 1)
        if self.noise:                       # random: must be sent, never drawn again
            mask = mask + torch.rand_like(mask) * 0.1
        h = self.emb(ids) + self.pos(pos)
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(h).transpose(1, 2)


def _decoder_run(cuts, noise=False, recompute=True, monkeypatch=None):
    torch.manual_seed(0)
    base = TinyDecoder(noise=noise)
    ids, y = torch.randint(0, 50, (4, 12)), torch.randint(0, 50, (4, 12))
    ref = copy.deepcopy(base)
    opt = torch.optim.SGD(ref.parameters(), lr=0.1)
    torch.manual_seed(1)
    for _ in range(3):
        opt.zero_grad()
        F.cross_entropy(ref(ids), y).backward()
        opt.step()
    model = copy.deepcopy(base)
    s = ts.slice(model, cuts=cuts, multi_tensor=True)
    torch.manual_seed(1)
    hist = s.train([(ids, y)] * 3, {"name": "SGD", "params": {"lr": 0.1}}, CE, epochs=1)
    _same(model, ref)
    return s, hist


def test_a_causal_mask_is_rebuilt_after_the_cut_not_sent():
    s, hist = _decoder_run(["add_2", "add_4"])
    names = s.graph.names()
    assert "triu.cut0" in names and "triu.cut1" in names        # rebuilt in slices 1 and 2
    send = s.graph.nodes[s.partitions[0].layer_indices[-1]]
    sent = [s.graph.nodes[p].name for p in send.predecessors]
    assert sent == ["add_2", "size", "getattr_2"]                # h, T and the device only
    h_bytes = 3 * 4 * 12 * 16 * 4
    assert all(h_bytes < b < 1.25 * h_bytes for b in hist[0]["bytes_fwd"])


def test_random_values_are_sent_not_drawn_again():
    s, _ = _decoder_run(["add_2"], noise=True)                  # exact: the noise crossed
    send = s.graph.nodes[s.partitions[0].layer_indices[-1]]
    assert "add" in [s.graph.nodes[p].name for p in send.predecessors]      # mask + noise
    assert not any(n.endswith(".cut0") for n in s.graph.names())


def test_without_recompute_the_mask_crosses():
    from torchslicer.core.multicut import pack_cuts
    g = ts.trace(TinyDecoder())
    packed, _ = pack_cuts(g, [g.index_of("add_2")], recompute=False)
    send = next(n for n in packed.nodes if isinstance(n.module, _SendCut))
    assert "triu" in [packed.nodes[p].name for p in send.predecessors]
