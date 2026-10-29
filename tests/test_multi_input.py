"""Models with several inputs (core/inputs.py): a batch's inputs are a tuple or dict of
tensors; the graph starts with one ``input.<name>`` node per tensor, all in slice 0, and
an input read later crosses the cuts packed. Each case is checked against the unsplit
model, on every tracer, locally, with micro-batches, on clusters, and after restore."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchslicer as ts
from torchslicer.config import RunConfig
from torchslicer.core.hybrid import _MethodCall
from torchslicer.core.inputs import _Input, describe, example, input_leaves, split_inputs
from torchslicer.core.model_graph import TraceError

SGD = {"name": "SGD", "params": {"lr": 0.1}}
CE = {"name": "CrossEntropyLoss"}
V, T = 40, 6


class Block(nn.Module):
    """Self-attention that reads a padding mask (1 = keep), then an MLP."""

    def __init__(self, d):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, h, keep):
        q, k, v = self.qkv(self.ln1(h)).chunk(3, -1)
        att = (q @ k.transpose(1, 2)) / q.size(-1) ** 0.5
        att = att.masked_fill(keep[:, None, :] == 0, float("-inf")).softmax(-1)
        h = h + self.o(att @ v)
        return h + self.ff(self.ln2(h))


class TinyVLM(nn.Module):
    """A CNN encodes the image into 16 tokens, projected and put before the text; every
    block reads the padding mask."""

    def __init__(self, d=16):
        super().__init__()
        self.vision = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(8, d, 3, stride=2, padding=1), nn.ReLU())
        self.proj = nn.Linear(d, d)
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(2))
        self.head = nn.Linear(d, V)

    def forward(self, input_ids, pixel_values, attention_mask):
        img = self.vision(pixel_values).flatten(2).transpose(1, 2)
        h = torch.cat([self.proj(img), self.emb(input_ids)], 1)
        keep = torch.cat([attention_mask.new_ones(attention_mask.size(0), img.size(1)), attention_mask], 1)
        for b in self.blocks:
            h = b(h, keep)
        return self.head(h[:, img.size(1):]).transpose(1, 2)


def _data(n=8, batch=4, seed=3, kind="tuple"):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, V, (n, T), generator=g)
    pix = torch.randn(n, 3, 8, 8, generator=g)
    mask = torch.ones(n, T, dtype=torch.long)
    mask[::2, -2:] = 0
    y = torch.randint(0, V, (n, T), generator=g)
    out = []
    for i in range(0, n, batch):
        a, b, c = ids[i:i + batch], pix[i:i + batch], mask[i:i + batch]
        x = (a, b, c) if kind == "tuple" else {"input_ids": a, "pixel_values": b, "attention_mask": c}
        out.append((x, y[i:i + batch]))
    return out


def _call(model, x):
    return model(*x) if isinstance(x, tuple) else model(**x)


def _fresh(make=TinyVLM):
    torch.manual_seed(0)
    return make().train()


def _unsplit(model, data, seed=1):
    torch.manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), **SGD["params"])
    for x, y in data:
        opt.zero_grad()
        F.cross_entropy(_call(model, x), y).backward()
        opt.step()
    return model


def _train(sliced, data, seed=1):
    torch.manual_seed(seed)
    return sliced.train(data, SGD, CE, epochs=1, verbose=False)


def _same(a, b, atol=1e-6):
    for (name, x), y in zip(a.state_dict().items(), b.state_dict().values()):
        assert torch.allclose(x.float(), y.float(), atol=atol), name


# ── the batch's inputs ─────────────────────────────────────────────────────────

def test_inputs_are_named_by_the_batch_structure():
    a, b, c = torch.zeros(2, 3), torch.ones(2), torch.zeros(2, 1)
    assert [n for n, _ in input_leaves((a, b))] == ["0", "1"]
    assert [n for n, _ in input_leaves({"x": a, "extra": [b, None, c]})] == ["x", "extra.0", "extra.2"]
    assert [n for n, _ in input_leaves((a, {"pixel_values": b}))] == ["0", "pixel_values"]
    main, aux = split_inputs({"pixel_values": b, "input_ids": a})
    assert main is a and set(aux) == {"pixel_values"}
    main, aux = split_inputs((a, b, c))
    assert main is a and list(aux) == ["1", "2"]
    assert split_inputs(a) == (a, {})


def test_recorded_inputs_rebuild_an_example():
    ids, pix = torch.randint(0, 9, (2, 3)), torch.randn(2, 3, 4)
    back = example(describe({"ids": ids, "pix": pix, "flag": True, "pair": (ids, None)}))
    assert torch.equal(back["ids"], ids) and back["pix"].shape == pix.shape
    assert back["flag"] is True and torch.equal(back["pair"][0], ids) and back["pair"][1] is None


# ── tracing and training ───────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["tuple", "dict"])
@pytest.mark.parametrize("tracer", ["auto", "fx", "export"])
def test_several_inputs_train_like_unsplit(tracer, kind):
    data = _data(kind=kind)
    ref = _unsplit(_fresh(), data)
    model = _fresh()
    sliced = ts.slice(model, n=3, tracer=tracer, example_input=data[0][0])
    names = ["0", "1", "2"] if kind == "tuple" else ["input_ids", "pixel_values", "attention_mask"]
    graph = sliced.graph
    assert graph.inputs == names
    assert [n.name for n in graph.nodes[:3]] == [f"input.{n}" for n in names]
    assert all(isinstance(n.module, _Input) for n in graph.nodes[:3])
    assert all(i in sliced.partitions[0].layer_indices for i in range(3))
    assert graph.source == ("exported" if tracer == "export" else "traced")
    _train(sliced, data)
    _same(model, ref)


def test_the_mask_crosses_every_cut_inside_the_language_model():
    data = _data()
    ref = _unsplit(_fresh(), data)
    model = _fresh()
    sliced = ts.slice(model, cuts=["vision.1", "blocks.0"], expand=["vision"], example_input=data[0][0])
    sends = [n for n in sliced.graph.nodes if n.name.endswith(".send")]
    assert len(sends) == 2              # the image features + the ids + the mask; then h + the mask
    _train(sliced, data)
    _same(model, ref)


def test_micro_batches():
    data = _data()
    ref = _unsplit(_fresh(), data)
    model = _fresh()
    sliced = ts.slice(model, n=3, example_input=data[0][0])
    torch.manual_seed(1)
    sliced.train(data, SGD, CE, epochs=1, use_gpipe=True, n_micro_batches=2)
    _same(model, ref, atol=1e-5)


def test_cost_strategies_and_summary_take_the_batch():
    x = _data(kind="dict")[0][0]
    sliced = ts.slice(_fresh(), strategy="flops", n=2, example_input=x)
    assert "input.pixel_values" in sliced.graph.summary(example_input=x)


# ── errors ─────────────────────────────────────────────────────────────────────

def test_without_an_example_fx_says_how_to_trace():
    with pytest.raises(TraceError, match="example_input="):
        ts.trace(_fresh(), tracer="fx")


def test_a_batch_without_an_input_the_graph_reads():
    data = _data(kind="dict")
    sliced = ts.slice(_fresh(), n=2, example_input=data[0][0])
    x, y = data[0]
    with pytest.raises(KeyError, match="attention_mask"):
        _train(sliced, [({k: v for k, v in x.items() if k != "attention_mask"}, y)])


# ── what export can't trace ────────────────────────────────────────────────────

class Positions(nn.Module):
    """Positions restarting after each separator token: a Python loop over the ids'
    values, in a method (as Qwen2-VL's ``get_rope_index``), and an attribute set in forward."""

    def __init__(self, d=16, sep=0):
        super().__init__()
        self.sep = sep
        self.emb, self.pos = nn.Embedding(V, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(2))
        self.head = nn.Linear(d, V)
        self.last = None

    def positions(self, ids):
        out = []
        for row in ids.tolist():
            p, k = [], 0
            for t in row:
                k = 0 if t == self.sep else k
                p.append(k)
                k += 1
            out.append(p)
        return torch.tensor(out, device=ids.device)

    def forward(self, input_ids, attention_mask):
        pos = self.positions(input_ids)
        self.last = pos
        h = self.emb(input_ids) + self.pos(pos)
        for b in self.blocks:
            h = b(h, attention_mask)
        return self.head(h).transpose(1, 2)


def _ids_data(n=8, batch=4):
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(0, V, (n, T), generator=g)
    ids[:, 2] = 0
    mask = torch.ones(n, T, dtype=torch.long)
    y = torch.randint(0, V, (n, T), generator=g)
    return [({"input_ids": ids[i:i + batch], "attention_mask": mask[i:i + batch]}, y[i:i + batch])
            for i in range(0, n, batch)]


def test_a_method_export_cannot_trace_is_one_call():
    data = _ids_data()
    ref = _unsplit(_fresh(Positions), data)
    model = _fresh(Positions)
    sliced = ts.slice(model, n=3, tracer="export", example_input=data[0][0])
    calls = [n for n in sliced.graph.nodes if isinstance(getattr(n.module, "target", None), _MethodCall)]
    assert [n.name for n in calls] == ["positions"]
    assert not list(calls[0].module.parameters())          # a copy of the module without weights
    assert sliced.graph.source == "mixed"
    assert not isinstance(model.last, torch._subclasses.FakeTensor)    # tracing left it alone
    _train(sliced, data)
    _same(model, ref)


class Weighted(Positions):
    """The same method, but it reads a weight: the module it belongs to stays whole instead."""

    def positions(self, ids):
        return super().positions(ids) + 0 * self.head.weight.sum().long()


def test_a_method_that_reads_weights_keeps_its_module_whole():
    data = _ids_data()
    with pytest.raises(TraceError):          # the module is the root: nothing else opens it
        ts.trace(_fresh(Weighted), tracer="export", example_input=data[0][0])


class Remembers(nn.Module):
    """Sets an attribute in forward: tracing must not leave a proxy or fake tensor there."""

    def __init__(self):
        super().__init__()
        self.a, self.b, self.last = nn.Linear(4, 4), nn.Linear(4, 4), None

    def forward(self, x):
        h = self.a(x)
        self.last = h
        return self.b(h)


@pytest.mark.parametrize("tracer", ["fx", "export"])
def test_tracing_leaves_attributes_set_in_forward_alone(tracer):
    model = Remembers()
    ts.trace(model, tracer=tracer, example_input=torch.randn(3, 4))
    assert model.last is None or type(model.last) is torch.Tensor


class SplitAt(nn.Module):
    """Splits at a position read from the data, and returns a tuple: neither tracer opens it."""

    def forward(self, h):
        k = int(h[0, 0].abs().argmax()) % h.size(1) + 1
        return h[:, :k].sum(1), h[:, k - 1:].mean(1)


class TwoHeads(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.inp, self.split, self.out = nn.Linear(d, d), SplitAt(), nn.Linear(d, 3)

    def forward(self, x, scale):
        a, b = self.split(self.inp(x) * scale)
        return self.out(a + b)


def test_no_cut_between_a_kept_call_and_the_items_of_its_tuple():
    g = torch.Generator().manual_seed(0)
    x, scale = torch.randn(4, 5, 8, generator=g), torch.rand(4, 1, 1, generator=g)
    data = [((x, scale), torch.randint(0, 3, (4,), generator=g))]
    ref = _unsplit(_fresh(TwoHeads), data)
    model = _fresh(TwoHeads)
    graph = ts.trace(model, tracer="export", example_input=(x, scale))
    kept = next(n for n in graph.nodes if n.name == "split")
    assert set(range(kept.index, max(kept.successors))) <= graph.no_cut
    sliced = ts.slice(graph, n=len(graph.multi_tensor_cuts()) + 1)     # every cut there can be
    _train(sliced, data)
    _same(model, ref)


# ── clusters and restore ───────────────────────────────────────────────────────

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
    ("centralized", "pipeline", "grpc"),    # the inputs go to a remote slice 0
    ("p2p", "vanilla", "tcp"),
])
def test_cluster_trains_like_unsplit(topology, setting, transport):
    data = _data(kind="dict")
    ref = _unsplit(_fresh(), data)
    model = _fresh()
    ts.run(lambda: model, lambda: data, run_config=_cfg(topology, setting, transport),
           verbose=False, example_input=data[0][0])
    _same(model, ref, atol=1e-5)


@pytest.mark.parametrize("kind", ["tuple", "dict"])
def test_restore_traces_the_same_inputs_again(tmp_path, kind):
    from test_p2p import _cfg as p2p_cfg, _driver, _followers
    data = _data(kind=kind)
    cfg = p2p_cfg(epochs=2)
    cfg.checkpoint.enabled, cfg.checkpoint.dir = True, str(tmp_path)
    followers = _followers(cfg, 2)
    try:
        model = _fresh()
        ts.slice(model, n=3, example_input=data[0][0],
                 executor=_driver(cfg, followers)).train(data, run_config=cfg)
        restored = ts.restore_model_from_run(_fresh(), str(tmp_path / cfg.run_id))
        _same(restored, model, atol=0)
    finally:
        for w in followers:
            w.stop()


# ── Hugging Face vision-language models ────────────────────────────────────────

class _Logits(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, **kw):
        return self.m(**kw).logits.transpose(1, 2)


def _llava(tr):
    vis = tr.CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                              num_attention_heads=2, image_size=32, patch_size=8)
    txt = tr.LlamaConfig(vocab_size=120, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2)
    m = tr.LlavaForConditionalGeneration(tr.LlavaConfig(
        vision_config=vis, text_config=txt, image_token_index=119, vision_feature_layer=-1,
        vision_feature_select_strategy="full"))
    ids = torch.randint(5, 119, (2, 25))
    ids[:, 1:18] = 119                       # 16 patches + the class token
    return m, {"input_ids": ids, "pixel_values": torch.randn(2, 3, 32, 32),
               "attention_mask": torch.ones_like(ids)}, 120


def _qwen2_vl(tr):
    c = tr.Qwen2VLConfig(
        vision_config=dict(depth=2, embed_dim=32, hidden_size=32, num_heads=2, mlp_ratio=2, patch_size=4,
                           spatial_merge_size=2, temporal_patch_size=2, in_channels=3),
        text_config=dict(vocab_size=200, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2,
                         rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]}),
        image_token_id=199, video_token_id=198, vision_start_token_id=197)
    m = tr.Qwen2VLForConditionalGeneration(c)
    ids = torch.randint(0, 190, (2, 14))
    ids[:, 1], ids[:, 2:6] = 197, 199        # one image: 4x4 patches, merged 2x2 -> 4 tokens
    return m, {"input_ids": ids, "pixel_values": torch.randn(32, 96),
               "image_grid_thw": torch.tensor([[1, 4, 4]] * 2), "attention_mask": torch.ones_like(ids),
               "mm_token_type_ids": (ids == 199).long()}, 200


@pytest.mark.parametrize("make", [_llava, _qwen2_vl])
def test_hugging_face_vision_language_models(make):
    tr = pytest.importorskip("transformers")
    tr.logging.set_verbosity_error()
    torch.manual_seed(0)
    m, x, vocab = make(tr)
    y = torch.randint(0, vocab, x["input_ids"].shape)
    model = _Logits(m).train()
    ref = _unsplit(_Logits(__import__("copy").deepcopy(m)).train(), [(x, y)])
    sliced = ts.slice(model, n=3, tracer="export", example_input=x)
    assert sliced.graph.inputs == list(x)
    _train(sliced, [(x, y)])
    _same(model, ref)
