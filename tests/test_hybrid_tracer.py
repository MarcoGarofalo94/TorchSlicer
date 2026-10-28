"""torch.fx and torch.export together, module by module (core/hybrid.py).

Each module is opened by the preferred tracer (fx for ``tracer="auto"``, export for
``tracer="export"``), else by the other one, else kept whole. The invariant: whatever the
mix, the sliced model trains exactly like the unsplit one, and cuts can fall inside every
module some tracer could open.
"""
import pytest
import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F

import torchslicer as ts
from torchslicer.config import RunConfig
from torchslicer.core.export_graph import _OpWrapper
from torchslicer.core.hybrid import _OpaqueCall

SGD = {"name": "SGD", "params": {"lr": 0.05, "momentum": 0.9}}
CE = {"name": "CrossEntropyLoss"}
V, D = 40, 16


# ── models ─────────────────────────────────────────────────────────────────────

class Windowed(nn.Module):
    """``if T > 8``: fx can't trace a branch on a size, export can."""

    def __init__(self):
        super().__init__()
        self.ln, self.l1, self.l2 = nn.LayerNorm(D), nn.Linear(D, 2 * D), nn.Linear(2 * D, D)

    def forward(self, h):
        T = h.shape[1]
        if T > 8:
            h = h + F.pad(h, (0, 0, 1, 0))[:, :T]
        return h + self.l2(F.gelu(self.l1(self.ln(h))))


class Mixed(nn.Module):
    """Two blocks only export can open, then nn.TransformerEncoder with a causal mask, which
    export can't trace (it checks whether the mask's values are causal) and fx can't open."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.mid = nn.Sequential(Windowed(), Windowed())
        layer = nn.TransformerEncoderLayer(D, 2, 2 * D, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(D, V)

    def forward(self, ids):
        T = ids.size(1)
        mask = torch.full((T, T), float("-inf"), device=ids.device).triu(1)
        return self.head(self.enc(self.mid(self.emb(ids)), mask=mask)).transpose(1, 2)


@torch.fx.wrap
def _gate(h):
    """Reads a value: export can't trace it, fx keeps it as one call."""
    return h * (1.0 if float(h.abs().amax()) > 0 else 0.0)


class Gated(nn.Module):
    """fx can trace this one (``_gate`` is a leaf for fx), export can't."""

    def __init__(self):
        super().__init__()
        self.inner, self.lin = Windowed(), nn.Linear(D, D)

    def forward(self, h):
        return _gate(self.lin(self.inner(h)))


class Outer(nn.Module):
    """fx can't trace this one (a branch on a size): exported, with Gated inside."""

    def __init__(self):
        super().__init__()
        self.pre, self.gated, self.post = nn.Linear(D, D), Gated(), nn.Linear(D, D)

    def forward(self, h):
        if h.shape[1] > 4:
            h = torch.relu(self.pre(h))
        return self.post(self.gated(h))


class Deep(nn.Module):
    """fx -> export (outer) -> fx (outer.gated) -> export (outer.gated.inner)."""

    def __init__(self):
        super().__init__()
        self.emb, self.outer, self.head = nn.Embedding(V, D), Outer(), nn.Linear(D, V)

    def forward(self, ids):
        return self.head(self.outer(self.emb(ids))).transpose(1, 2)


class Pair(nn.Module):
    """Returns a tuple, takes an int and keywords: fx can't open it (size branch)."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D, D)

    def forward(self, h, T, scale=1.0, **kwargs):
        if h.shape[1] == T:
            h = self.lin(h) * scale
        return h, h.mean(1)


class Structured(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.pair, self.shared, self.head = nn.Embedding(V, D), Pair(), Windowed(), nn.Linear(D, V)

    def forward(self, ids):
        out = self.pair(self.emb(ids), ids.size(1), scale=2.0)
        h = self.shared(self.shared(out[0])) + out[1][:, None]
        return self.head(h).transpose(1, 2)


class Growing(nn.Module):
    """Doubles the length: an output size no input size explains."""

    def forward(self, h):
        if float(h.sum()) != 0.5:            # export can't trace it
            h = h.repeat(1, 2, 1)
        return h


class Grows(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.grow, self.head = nn.Embedding(V, D), Growing(), nn.Linear(D, V)

    def forward(self, ids):
        h = self.grow(self.emb(ids))
        return self.head(h.reshape(h.shape[0], -1, D)).transpose(1, 2)


def _data(n=16, T=12, batch=4, seed=7):
    g = torch.Generator().manual_seed(seed)
    x, y = torch.randint(0, V, (n, T), generator=g), torch.randint(0, V, (n, T), generator=g)
    return [(x[i:i + batch], y[i:i + batch]) for i in range(0, n, batch)]


EXAMPLE = _data()[0][0]


def _fresh(make):
    torch.manual_seed(0)
    return make().train()


def _unsplit(model, data, seed=1):
    torch.manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), **SGD["params"])
    for x, y in data:
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
    return model


def _train(sliced, data, seed=1):
    torch.manual_seed(seed)
    return sliced.train(data, SGD, CE, epochs=1)


def _same(a, b, atol=1e-6):
    for (name, x), y in zip(a.state_dict().items(), b.state_dict().values()):
        assert torch.allclose(x.float(), y.float(), atol=atol), name


def _in(graph, path):
    """Nodes inside the module ``path``."""
    return [n for n in graph.nodes if path in n.scope or n.name == path]


# ── what each tracer opens ─────────────────────────────────────────────────────

def test_auto_exports_the_modules_fx_cannot_open():
    g = ts.trace(_fresh(Mixed), expand="all", example_input=EXAMPLE)
    assert g.source == "mixed"
    for block in ("mid.0", "mid.1"):           # opened by export: ATen ops
        assert any(isinstance(n.module, _OpWrapper) for n in _in(g, block))
    enc = [n for n in g.nodes if n.name == "enc"]
    assert len(enc) == 1                       # neither tracer opens it: one node
    assert not isinstance(enc[0].module, _OpaqueCall)   # fx made it, so no inferred shapes
    assert g.export_input["prefer"] == "fx"


def test_export_keeps_whole_the_modules_neither_can_open():
    g = ts.trace(_fresh(Mixed), tracer="export", example_input=EXAMPLE)
    enc = [n for n in g.nodes if n.name == "enc"]
    assert len(enc) == 1 and isinstance(enc[0].module, _OpaqueCall)
    assert all(isinstance(n.module, _OpWrapper) or n.name == "enc" or n.module.__class__.__name__
               in ("_AttrWrapper", "_InputGuard") for n in g.nodes)


def test_fx_alone_still_refuses():
    with pytest.raises(ts.TraceError):
        ts.trace(_fresh(Mixed), expand="all", tracer="fx")


@pytest.mark.parametrize("tracer", ["auto", "export"])
def test_tracers_alternate_as_deep_as_needed(tracer):
    g = ts.trace(_fresh(Deep), expand="all", tracer=tracer, example_input=EXAMPLE)
    # outer: export (fx fails on its branch); outer.gated: fx (export fails on _gate);
    # outer.gated.inner: export again (fx fails on its branch)
    gated = [n for n in _in(g, "outer.gated") if "outer.gated.inner" not in n.scope]
    assert any(n.name == "outer.gated.lin" for n in gated)          # an fx module node
    assert any(isinstance(n.module, _OpWrapper) for n in _in(g, "outer.gated.inner"))
    assert any(isinstance(n.module, _OpWrapper) for n in _in(g, "outer")
               if "outer.gated" not in n.scope)
    x = EXAMPLE
    torch.testing.assert_close(g.run_nodes(x)[-1], _fresh(Deep)(x), rtol=0, atol=0)


# ── training equals unsplit ────────────────────────────────────────────────────

@pytest.mark.parametrize("make", [Mixed, Deep, Structured])
@pytest.mark.parametrize("tracer", ["auto", "export"])
@pytest.mark.parametrize("n", [2, 4])
def test_mixed_graphs_train_like_unsplit(make, tracer, n):
    data = _data()
    ref = _unsplit(_fresh(make), data)
    model = _fresh(make)
    sliced = ts.slice(model, n=n, multi_tensor=True, expand="all", tracer=tracer, example_input=EXAMPLE)
    _train(sliced, data)
    _same(model, ref)


def test_a_cut_inside_an_exported_block():
    data = _data()
    ref = _unsplit(_fresh(Mixed), data)
    model = _fresh(Mixed)
    sliced = ts.slice(model, cuts=["mid.0", "mid.1.l1"], multi_tensor=True, expand="all",
                      example_input=EXAMPLE)
    after = sliced.graph.base.nodes[sliced.graph.base_cuts[1]]
    assert "mid.1" in after.scope and "mid.1.l2" not in after.scope    # inside mid.1, before l2
    _train(sliced, data)
    _same(model, ref)


def test_cuts_send_the_hidden_state():
    sliced = ts.slice(_fresh(Mixed), n=3, multi_tensor=True, expand="all", example_input=EXAMPLE)
    outs = sliced.graph._output_sizes(EXAMPLE, None)
    hidden = EXAMPLE.numel() * D * 4
    for p in sliced.partitions[:-1]:
        assert outs[p.layer_indices[-1]][1] < 1.5 * hidden


def test_structured_outputs_ints_and_shared_modules():
    g = ts.trace(_fresh(Structured), tracer="auto", expand="all", example_input=EXAMPLE)
    assert {"shared", "shared:1"} <= {s for n in g.nodes for s in n.scope}
    for T in (12, 10):                        # the int input stays symbolic
        x = torch.randint(0, V, (3, T))
        torch.testing.assert_close(g.run_nodes(x)[-1], _fresh(Structured)(x), rtol=0, atol=0)
    # shared's export assumed T > 8 (its branch): a shorter input is refused, not run wrong
    with pytest.raises(ValueError, match=r"input 0 of shared dim 1 is 5.*\[9, None\]"):
        g.run_nodes(torch.randint(0, V, (3, 5)))


def test_inferred_shapes_are_checked_when_the_input_changes():
    g = ts.trace(_fresh(Grows), tracer="export", example_input=EXAMPLE)
    torch.testing.assert_close(g.run_nodes(EXAMPLE)[-1], _fresh(Grows)(EXAMPLE), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="trace again"):
        g.run_nodes(torch.randint(0, V, (4, 7)))


def test_eval_and_the_model_are_left_alone():
    model = _fresh(Deep)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    state = torch.random.get_rng_state()
    model.eval()
    ts.trace(model, tracer="export", expand="all", example_input=EXAMPLE)
    assert not model.training and torch.equal(state, torch.random.get_rng_state())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert "forward" not in vars(model.outer.gated)      # the opaque patch is removed


# ── executors and restore ──────────────────────────────────────────────────────

def _cfg(topology, setting, transport):
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
    ("p2p", "vanilla", "tcp"),
])
def test_cluster_trains_like_unsplit(topology, setting, transport):
    data = _data()
    ref = _unsplit(_fresh(Deep), data)
    model = _fresh(Deep)
    ts.run(lambda: model, lambda: data, run_config=_cfg(topology, setting, transport),
           verbose=False, multi_tensor=True, expand="all", tracer="export", example_input=EXAMPLE)
    _same(model, ref, atol=1e-5)


@pytest.mark.parametrize("tracer", ["auto", "export"])
def test_restore_traces_the_same_mix_again(tmp_path, tracer):
    from test_p2p import _cfg as p2p_cfg, _driver, _followers
    data = _data()
    cfg = p2p_cfg(epochs=2)
    cfg.checkpoint.enabled, cfg.checkpoint.dir = True, str(tmp_path)
    followers = _followers(cfg, 2)
    try:
        model = _fresh(Mixed)
        ts.slice(model, n=3, multi_tensor=True, expand="all", tracer=tracer, example_input=EXAMPLE,
                 executor=_driver(cfg, followers)).train(data, run_config=cfg)
        restored = ts.restore_model_from_run(_fresh(Mixed), str(tmp_path / cfg.run_id))
        _same(restored, model, atol=0)
    finally:
        for w in followers:
            w.stop()
