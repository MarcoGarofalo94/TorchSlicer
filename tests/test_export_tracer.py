"""M17 ``torch.export`` tracer, and tensor factories called with traced sizes under fx.

The invariant: a model traced with ``tracer="export"`` trains exactly like the unsplit one,
in every executor, and its cuts send what crosses between blocks, not the inside of a layer.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchslicer as ts
from torchslicer.config import RunConfig
from torchslicer.core.export_graph import _InputGuard, _OpWrapper

SGD = {"name": "SGD", "params": {"lr": 0.05, "momentum": 0.9}}
CE = {"name": "CrossEntropyLoss"}


# ── models ─────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, 2, dropout=p, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(p)

    def forward(self, h, mask):
        a = self.ln1(h)
        h = h + self.drop(self.attn(a, a, a, attn_mask=mask, need_weights=False)[0])
        return h + self.drop(self.ff(self.ln2(h)))


class TinyLM(nn.Module):
    """What torch.fx can't trace: a branch on the input's length, and a dict output."""

    def __init__(self, vocab=40, d=16, n=3, p=0.1):
        super().__init__()
        self.emb, self.pos = nn.Embedding(vocab, d), nn.Embedding(64, d)
        self.blocks = nn.ModuleList(Block(d, p) for _ in range(n))
        self.drop = nn.Dropout(p)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        T = ids.shape[1]
        mask = torch.full((T, T), float("-inf"), device=ids.device).triu(1) if T > 1 else None
        h = self.drop(self.emb(ids) + self.pos(torch.arange(T, device=ids.device)))
        for blk in self.blocks:
            h = blk(h, mask)
        return {"logits": self.head(h).transpose(1, 2), "hidden": h}


class OnesLM(nn.Module):
    """``torch.ones(T, T)`` with ``T`` read from the input: fx could not trace it."""

    def __init__(self, vocab=40, d=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.l1, self.l2, self.head = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, vocab)

    def forward(self, ids):
        T = ids.size(1)
        causal = torch.ones(T, T, device=ids.device).tril() / torch.arange(1, T + 1, device=ids.device)[:, None]
        h = self.emb(ids)
        h = h + causal @ torch.relu(self.l1(h))
        h = h + causal @ torch.relu(self.l2(h))
        return self.head(h).transpose(1, 2)


class Masked(nn.Module):
    """A tensor written in place (``m[:, :4] = 1``) and read again later, as swin's mask."""

    def __init__(self):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 3)

    def forward(self, x):
        m = x.new_zeros(x.shape[0], 8)
        m[:, :4] = 1.0
        h = self.l2(torch.relu(self.l1(x) * m)) + m
        return self.l3(torch.relu(h))


def _data(n=16, T=12, vocab=40, batch=4, seed=7):
    g = torch.Generator().manual_seed(seed)
    x, y = torch.randint(0, vocab, (n, T), generator=g), torch.randint(0, vocab, (n, T), generator=g)
    return [(x[i:i + batch], y[i:i + batch]) for i in range(0, n, batch)]


def _fresh(make=TinyLM):
    torch.manual_seed(0)
    return make().train()


def _logits(out):
    return out["logits"] if isinstance(out, dict) else out


def _unsplit(model, data, epochs=1, seed=1):
    torch.manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), **SGD["params"])
    for _ in range(epochs):
        for x, y in data:
            opt.zero_grad()
            F.cross_entropy(_logits(model(x)), y).backward()
            opt.step()
    return model


def _train(sliced, data, seed=1, **kw):
    torch.manual_seed(seed)
    return sliced.train(data, SGD, CE, epochs=1, **kw)


def _same(a, b, atol=1e-6):
    for (name, x), y in zip(a.state_dict().items(), b.state_dict().values()):
        assert torch.allclose(x.float(), y.float(), atol=atol), name


EXAMPLE = _data()[0][0]


# ── tensor factories under fx ──────────────────────────────────────────────────

def test_factories_called_with_traced_sizes_become_graph_nodes():
    graph = ts.trace(OnesLM())
    assert {"ones", "arange"} <= set(graph.names())
    data = _data()
    ref = _unsplit(_fresh(OnesLM), data)
    model = _fresh(OnesLM)
    sliced = ts.slice(model, n=3, multi_tensor=True)
    # the causal matrix is rebuilt after each cut from the length, not sent
    assert any(name.startswith("ones.cut") for name in sliced.graph.names())
    _train(sliced, data)
    _same(model, ref)
    assert torch.ones(2, 2).sum() == 4             # the real factories are back


# ── choosing the tracer ────────────────────────────────────────────────────────

def test_auto_falls_back_to_export_when_fx_fails_and_an_example_input_is_given():
    with pytest.raises(ts.TraceError, match="example_input="):
        ts.trace(TinyLM())
    assert ts.trace(TinyLM(), example_input=EXAMPLE).source == "exported"
    with pytest.raises(ts.TraceError):
        ts.trace(TinyLM(), tracer="fx", example_input=EXAMPLE)
    assert ts.trace(OnesLM(), example_input=EXAMPLE).source == "traced"   # fx works: fx
    assert ts.trace(OnesLM(), tracer="export", example_input=EXAMPLE).source == "exported"
    with pytest.raises(ValueError, match="needs example_input"):
        ts.trace(TinyLM(), tracer="export")
    with pytest.raises(ValueError, match="batch of at least 2"):
        ts.trace(TinyLM(), tracer="export", example_input=EXAMPLE[:1])


def test_tracing_changes_neither_the_model_nor_the_random_state():
    tv = pytest.importorskip("torchvision")
    model = tv.models.resnet18(num_classes=10).train()
    x = torch.randn(2, 3, 32, 32)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.random.get_rng_state()
    ts.trace(model, tracer="export", example_input=x)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert model.training


def test_the_exported_graph_is_aten_ops_and_library_modules_only():
    graph = ts.trace(TinyLM(), tracer="export", example_input=EXAMPLE)
    assert isinstance(graph.nodes[0].module, _InputGuard)
    assert any(isinstance(n.module, _OpWrapper) for n in graph.nodes)
    # no class of the user's code: workers need nothing but torchslicer and torch
    assert all(type(m).__module__.split(".")[0] in ("torchslicer", "torch")
               for n in graph.nodes for m in n.module.modules())
    # tied and shared parameters stay the model's own objects
    own = {id(p) for p in TinyLM.parameters(graph.model)}
    assert all(id(p) in own for n in graph.nodes for p in n.module.parameters())


# ── exactness ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [2, 3, 4])
def test_exported_model_trains_like_unsplit_with_dropout(n):
    data = _data()
    ref = _unsplit(_fresh(), data)
    model = _fresh()
    _train(ts.slice(model, n=n, multi_tensor=True, example_input=EXAMPLE), data)
    _same(model, ref)


def test_gpipe_trains_like_unsplit():
    data = _data(batch=8)
    make = lambda: TinyLM(p=0.0)
    ref = _unsplit(_fresh(make), data)
    model = _fresh(make)
    _train(ts.slice(model, n=3, multi_tensor=True, example_input=EXAMPLE), data, n_micro_batches=2)
    _same(model, ref, atol=1e-5)


def test_cuts_land_between_layers_and_send_only_the_hidden_state():
    data = _data()
    sliced = ts.slice(_fresh(), n=3, multi_tensor=True, example_input=EXAMPLE)
    hidden = 4 * 12 * 16 * 4                          # [B, T, d] float32
    for sent in _train(sliced, data)[0]["bytes_fwd"]:
        assert sent / len(data) < 1.5 * hidden, sent   # + a header of sizes; inside the MLP: >= 2x


def test_named_cuts_use_module_names():
    sliced = ts.slice(_fresh(), cuts=["blocks.0", "blocks.1"], multi_tensor=True, example_input=EXAMPLE)
    base = sliced.graph.base
    last = [p.layer_indices[-1] for p in sliced.partitions[:-1]]
    names = [sliced.graph.nodes[i].name for i in last]
    assert all(name.startswith("cut") for name in names)
    assert [base.index_of(f"blocks.{k}") for k in (0, 1)] == sliced.graph.base_cuts
    data = _data()
    ref = _unsplit(_fresh(), data)
    model = _fresh()
    _train(ts.slice(model, cuts=["blocks.0", "blocks.1"], multi_tensor=True, example_input=EXAMPLE), data)
    _same(model, ref)


def test_eval_turns_dropout_off():
    model = _fresh()
    x = EXAMPLE
    layers = ts.slice(model, n=3, multi_tensor=True, example_input=x).slices()
    h = x
    for layer in layers:
        layer.eval()
        h = layer(h)
    model.eval()
    with torch.no_grad():
        assert torch.equal(h.detach(), model(x)["logits"])


def test_other_batch_sizes_and_lengths_run_and_fixed_sizes_are_checked():
    data = _data(n=12, T=9, batch=3)                  # exported on [4, 12]
    ref = _unsplit(_fresh(), data)
    model = _fresh()
    sliced = ts.slice(model, n=3, multi_tensor=True, example_input=EXAMPLE)
    _train(sliced, data)
    _same(model, ref)
    with pytest.raises(ValueError, match="2-d input"):
        sliced.slices()[0](torch.zeros(2, 3, 4, dtype=torch.long))


def test_a_tensor_written_in_place_is_never_cut_from_its_write():
    g = torch.Generator().manual_seed(5)
    x, y = torch.randn(16, 8, generator=g), torch.randint(0, 3, (16,), generator=g)
    data = [(x[i:i + 4], y[i:i + 4]) for i in range(0, 16, 4)]
    graph = ts.trace(Masked(), example_input=x[:4])          # fx can't trace the item assignment
    zeros, fill = graph.index_of("new_zeros"), graph.index_of("fill_")
    assert not set(range(zeros, fill)) & set(graph.multi_tensor_cuts() + graph.valid_cuts())
    for n in (2, 3, 4):
        ref = _unsplit(_fresh(Masked), data)
        model = _fresh(Masked)
        _train(ts.slice(model, n=n, multi_tensor=True, example_input=x[:4]), data)
        _same(model, ref)


# ── executors and restore ──────────────────────────────────────────────────────

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
    ("p2p", "vanilla", "tcp"),
])
def test_cluster_trains_like_unsplit(topology, setting, transport):
    make = lambda: TinyLM(p=0.0)
    data = _data()
    ref = _unsplit(_fresh(make), data)
    model = _fresh(make)
    ts.run(lambda: model, lambda: data, run_config=_cfg(topology, setting, transport),
           verbose=False, multi_tensor=True, example_input=EXAMPLE)
    _same(model, ref, atol=1e-5)


def test_restore_exports_the_model_again(tmp_path):
    from test_p2p import _cfg as p2p_cfg, _driver, _followers
    make = lambda: TinyLM(p=0.0)
    data = _data()
    cfg = p2p_cfg(epochs=2)
    cfg.checkpoint.enabled, cfg.checkpoint.dir = True, str(tmp_path)
    followers = _followers(cfg, 2)
    try:
        model = _fresh(make)
        ts.slice(model, n=3, multi_tensor=True, example_input=EXAMPLE,
                 executor=_driver(cfg, followers)).train(data, run_config=cfg)
        restored = ts.restore_model_from_run(_fresh(make), str(tmp_path / cfg.run_id))
        _same(restored, model, atol=0)
    finally:
        for w in followers:
            w.stop()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_exported_on_the_cpu_runs_on_cuda():
    make = lambda: TinyLM(p=0.0)
    data = _data()
    ref = _unsplit(_fresh(make).cuda(), [(x.cuda(), y.cuda()) for x, y in data])
    model = _fresh(make)
    sliced = ts.slice(model, n=3, multi_tensor=True, example_input=EXAMPLE)   # arange(device=cpu)
    cfg = RunConfig()
    cfg.network.device = "cuda"
    cfg.logging.enabled = False
    cfg.training.optimizer, cfg.training.criterion, cfg.training.epochs = SGD, CE, 1
    sliced.train(data, run_config=cfg)
    _same(model.cuda(), ref, atol=1e-5)


# ── real models ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("multi_tensor", [False, True])
def test_resnet_exports_with_batch_norm_statistics(multi_tensor):
    tv = pytest.importorskip("torchvision")
    make = lambda: tv.models.resnet18(num_classes=10)
    g = torch.Generator().manual_seed(3)
    x, y = torch.randn(8, 3, 32, 32, generator=g), torch.randint(0, 10, (8,), generator=g)
    ref = _unsplit(_fresh(make), [(x, y)])
    model = _fresh(make)
    kw = dict(n=4) if multi_tensor else dict(cuts=["layer1", "layer3.0", "avgpool"])
    sliced = ts.slice(model, tracer="export", example_input=x, multi_tensor=multi_tensor, **kw)
    _train(sliced, [(x, y)])
    _same(model, ref, atol=1e-5)                       # running statistics included
    before = {k: v.clone() for k, v in model.state_dict().items()}
    layers = sliced.slices()
    h = x
    for layer in layers:
        layer.eval()
        h = layer(h)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())   # eval: no update


@pytest.mark.parametrize("name", ["gpt2", "llama", "mixtral"])
def test_hugging_face_models_slice_without_pack(name):
    tr = pytest.importorskip("transformers")
    tr.logging.set_verbosity_error()

    def make():
        if name == "gpt2":
            m = tr.GPT2LMHeadModel(tr.GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=40,
                                                 bos_token_id=0, eos_token_id=0))
        elif name == "llama":
            m = tr.LlamaForCausalLM(tr.LlamaConfig(num_hidden_layers=2, hidden_size=32, intermediate_size=64,
                                                   num_attention_heads=2, num_key_value_heads=2, vocab_size=40))
        else:
            m = tr.MixtralForCausalLM(tr.MixtralConfig(num_hidden_layers=2, hidden_size=32, intermediate_size=64,
                                                       num_attention_heads=2, num_key_value_heads=2,
                                                       vocab_size=40, num_local_experts=4))
        return _LogitsFirst(m)

    data = _data()
    ref = _unsplit(_fresh(make), data)
    model = _fresh(make)
    history = _train(ts.slice(model, n=3, multi_tensor=True, tracer="export", example_input=EXAMPLE), data)
    _same(model, ref)
    assert max(history[0]["bytes_fwd"]) / len(data) < 1.5 * 4 * 12 * 32 * 4


class _LogitsFirst(nn.Module):
    """A Hugging Face causal LM whose logits are ``[B, vocab, T]``, for ``nn.CrossEntropyLoss``."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids):
        return self.model(ids).logits.transpose(1, 2)
