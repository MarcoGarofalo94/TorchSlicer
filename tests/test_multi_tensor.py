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
from torchslicer.core.multicut import _RecvCut, _SendCut, crossing, pack_cuts

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
