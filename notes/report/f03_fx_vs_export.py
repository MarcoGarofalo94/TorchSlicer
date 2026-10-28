"""F3 check: torch.fx vs torch.export as the tracer (tracer="fx" | "export" | "auto"), and mixing them.

Small models, CPU. Same model, two graphs: what each tracer builds, what it can trace, where the
cuts land, what crosses them, and whether training stays exact.
1. One model, two graphs: ResNet-18 under fx (leaves, expand="all") and export. Nodes, names,
   single-tensor cuts, trace time.
2. What each tracer can trace: a table of models x tracers (fx, export, auto).
3. Same named cuts, both graphs: what crosses, and training equals unsplit.
4. Language models (multi_tensor=True): fx vs export on a model fx can trace; GPT-2 with pack=
   vs export (no pack).
5. Mixing them, module by module: a model with blocks only export can open and a block only fx can
   trace. Which tracer opened which module, where cuts can go, training equals unsplit; then a
   model that alternates fx -> export -> fx -> export.

Run: .venv/bin/python notes/report/f03_fx_vs_export.py
"""
import copy
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import transformers

HERE = Path(globals().get("__file__", Path.cwd() / "_")).resolve().parent   # a notebook has no __file__
ROOT = next(p for p in [HERE, *HERE.parents] if (p / "examples" / "pack").is_dir())
sys.path.insert(0, str(ROOT))                                   # the repo root, for examples.pack
import torchslicer as ts
from examples.pack.gpt2 import pack_gpt2

logging.getLogger("torchslicer").setLevel(logging.WARNING)
transformers.logging.set_verbosity_error()
torch.manual_seed(0)

SGD = {"name": "SGD", "params": {"lr": 0.05, "momentum": 0.9}}
CE = {"name": "CrossEntropyLoss"}


def section(title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def fresh(base):
    return copy.deepcopy(base).train()


def max_diff(a, b):
    return max((x.float() - y.float()).abs().max().item()
               for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def logits(out):
    return out.logits.transpose(1, 2) if hasattr(out, "logits") else out


def train_unsplit(model, data, seed=1):
    torch.manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), **SGD["params"])
    for x, y in data:
        opt.zero_grad()
        F.cross_entropy(logits(model(x)), y).backward()
        opt.step()
    return model


def train_sliced(sliced, data, seed=1):
    torch.manual_seed(seed)
    return sliced.train(data, SGD, CE, epochs=1)


def cut_bytes(sliced, x):
    """Bytes crossing each cut on input x (the last node of every slice but the last)."""
    outs = sliced.graph._output_sizes(x, None)
    return [outs[p.layer_indices[-1]][1] for p in sliced.partitions[:-1]]


def kib(sizes):
    return "[" + ", ".join(f"{b / 1024:.1f}" for b in sizes) + "] KiB"


def short(err):
    msg = str(err).strip().splitlines()[0]
    return f"{type(err).__name__}: {msg[:90]}"


# ── models ─────────────────────────────────────────────────────────────────────

class LM(nn.Module):
    """A small causal LM fx can trace: the mask is built from the input's length."""

    def __init__(self, vocab=64, d=32, n=4):
        super().__init__()
        self.emb, self.pos = nn.Embedding(vocab, d), nn.Embedding(64, d)
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(d, 2, 2 * d, dropout=0.0, batch_first=True)
                                    for _ in range(n))
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        T = ids.size(1)
        mask = torch.full((T, T), float("-inf"), device=ids.device).triu(1)
        h = self.emb(ids) + self.pos(torch.arange(T, device=ids.device))
        for blk in self.blocks:
            h = blk(h, src_mask=mask)
        return self.head(h).transpose(1, 2)


class Windowed(nn.Module):
    """Branches on the input's length: fx can't trace `if T > 8`, export can."""

    def __init__(self, d=32):
        super().__init__()
        self.ln, self.l1, self.l2 = nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.Linear(2 * d, d)

    def forward(self, h):
        T = h.shape[1]
        if T > 8:                                    # long input: mix neighbours first
            h = h + F.pad(h, (0, 0, 1, 0))[:, :T]
        return h + self.l2(F.gelu(self.l1(self.ln(h))))


class Hybrid(nn.Module):
    """Embedding, two blocks only export can open (Windowed), then nn.TransformerEncoder with a
    causal mask: export can't trace it (PyTorch checks whether the mask's values are causal),
    fx can, as one node (fx can't open it either)."""

    def __init__(self, vocab=64, d=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.mid = nn.Sequential(Windowed(d), Windowed(d))
        layer = nn.TransformerEncoderLayer(d, 2, 2 * d, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        T = ids.size(1)
        mask = torch.full((T, T), float("-inf"), device=ids.device).triu(1)
        h = self.mid(self.emb(ids))
        return self.head(self.enc(h, mask=mask)).transpose(1, 2)


class CausalLM(nn.Module):
    """An HF causal LM returning logits as [B, V, T], what nn.CrossEntropyLoss takes."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids):
        return self.model(ids).logits.transpose(1, 2)


RESNET = torchvision.models.resnet18(num_classes=10)
IMG = torch.randn(4, 3, 32, 32)
IMG_DATA = [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))) for _ in range(3)]

VOCAB, T = 64, 16
IDS = torch.randint(0, VOCAB, (4, T))
TOK_DATA = [(torch.randint(0, VOCAB, (4, T)), torch.randint(0, VOCAB, (4, T))) for _ in range(3)]
GPT2 = transformers.GPT2LMHeadModel(transformers.GPT2Config(
    vocab_size=VOCAB, n_positions=32, n_embd=32, n_layer=4, n_head=2,
    resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0))


# ── 1. One model, two graphs ───────────────────────────────────────────────────
section("1. ResNet-18: the graph each tracer builds")
graphs = {}
for label, kw in [("fx (children are leaves)", dict(tracer="fx")),
                  ('fx, expand="all"', dict(tracer="fx", expand="all")),
                  ("export", dict(tracer="export", example_input=IMG))]:
    t0 = time.perf_counter()
    g = ts.trace(fresh(RESNET), **kw)
    graphs[label] = g
    print(f"{label:26s} {len(g):4d} nodes, {len(g.valid_cuts()):3d} single-tensor cuts, "
          f"traced in {time.perf_counter() - t0:.2f}s")

print("\nThe nodes of layer1.0 (one BasicBlock); export's parameter reads (_AttrWrapper) counted apart:")
for label, g in graphs.items():
    nodes = [n for n in g.nodes if n.name.startswith("layer1.0") or "layer1.0" in n.scope]
    ops = [n.name for n in nodes if type(n.module).__name__ != "_AttrWrapper"]
    attrs = len(nodes) - len(ops)
    print(f"  {label:26s} {ops if ops else '(inside the node layer1)'}" + (f" + {attrs} params/buffers" if attrs else ""))
print("\nWhat a node holds, export:")
for n in graphs["export"].nodes[1:8]:
    print(f"  {n.name:28s} {type(n.module).__name__:14s} scope={n.scope[-1:]}"
          f"  out={n.out_bytes / 1024:.1f} KiB")

# ── 2. What each tracer can trace ──────────────────────────────────────────────
section("2. What each tracer can trace (nodes, or the error)")
MODELS = {
    "resnet18": (RESNET, IMG),
    "LM (TransformerEncoderLayer + causal mask)": (LM(), IDS),
    "Hybrid (Windowed + nn.TransformerEncoder)": (Hybrid(), IDS),
    'Hybrid, expand="all"': (Hybrid(), IDS),
    "GPT-2 (HF, wrapped)": (CausalLM(GPT2), IDS),
}
for name, (base, x) in MODELS.items():
    print(f"\n{name}")
    for tracer in ["fx", "export", "auto"]:
        kw = dict(expand="all") if "expand" in name else {}
        try:
            g = ts.trace(fresh(base), tracer=tracer, example_input=x, **kw)
            print(f"  {tracer:7s} {len(g):5d} nodes ({g.source})")
        except Exception as e:
            print(f"  {tracer:7s} {short(e)}")

# ── 3. Same named cuts, both graphs ────────────────────────────────────────────
section('3. ResNet-18 cut after "layer1" and "layer3.0": what crosses, and training equals unsplit')
ref = train_unsplit(fresh(RESNET), IMG_DATA)
for tracer, kw in [("fx", {}), ("export", dict(example_input=IMG))]:
    m = fresh(RESNET)
    s = ts.slice(m, cuts=["layer1", "layer3.0"], tracer=tracer, **kw)
    train_sliced(s, IMG_DATA)
    print(f"  {tracer:7s} {[len(p.layer_indices) for p in s.partitions]} nodes per slice, "
          f"cuts send {kib(cut_bytes(s, IMG))}, max |w - unsplit| = {max_diff(m, ref):.1e}")

# ── 4. Language models ─────────────────────────────────────────────────────────
section("4a. LM (fx can trace it), n=3, multi_tensor=True")
ref = train_unsplit(fresh(LM_BASE := LM()), TOK_DATA)
for tracer, kw in [("fx", {}), ('fx, expand="all"', dict(expand="all")),
                   ("export", dict(example_input=IDS))]:
    m = fresh(LM_BASE)
    s = ts.slice(m, n=3, multi_tensor=True, tracer=tracer.split(",")[0], **kw)
    train_sliced(s, TOK_DATA)
    print(f"  {tracer:17s} {len(s.graph.base):4d} nodes, cuts send {kib(cut_bytes(s, IDS))}, "
          f"max |w - unsplit| = {max_diff(m, ref):.1e}")
print(f"  (the hidden state is {4 * T * 32 * 4 / 1024:.1f} KiB)")

section("4b. GPT-2: pack= (stages by hand) vs export (no pack), n=3")
ref = train_unsplit(fresh(GPT2), TOK_DATA)
for label, make in [("pack=pack_gpt2", lambda m: ts.slice(m, n=3, pack=pack_gpt2)),
                    ("export", lambda m: ts.slice(CausalLM(m), n=3, multi_tensor=True,
                                                   tracer="export", example_input=IDS))]:
    m = fresh(GPT2)
    s = make(m)
    train_sliced(s, TOK_DATA)
    base = getattr(s.graph, "base", None) or s.graph
    print(f"  {label:15s} {len(base):4d} nodes, cuts send {kib(cut_bytes(s, IDS))}, "
          f"max |w - unsplit| = {max_diff(m, ref):.1e}")

# ── 5. Mixing the tracers ──────────────────────────────────────────────────────
section("5a. Hybrid: Windowed needs export, nn.TransformerEncoder can't be opened by either")
from torchslicer.core.export_graph import _OpWrapper    # noqa: E402
from torchslicer.core.hybrid import _OpaqueCall, _check_input   # noqa: E402


def origin(node):
    """Which tracer made a node: an ATen op from export, a module/function call from fx, or a
    module kept whole."""
    m = node.module
    if isinstance(m, _OpaqueCall):
        return "kept whole (opaque in export)"
    if isinstance(m, _OpWrapper):
        return "export"
    if getattr(m, "target", None) is _check_input:
        return "size check"
    return "fx" if type(m).__module__.startswith(("torch.nn", "torchslicer.core.model_graph")) else "other"


def by_module(g, modules):
    rows = {}
    for path in modules:
        nodes = [n for n in g.nodes if path in n.scope or n.name == path]
        kinds = sorted({origin(n) for n in nodes if type(n.module).__name__ != "_AttrWrapper"})
        rows[path] = f"{len(nodes):3d} nodes, {', '.join(kinds)}"
    return rows


x, HYBRID = IDS, Hybrid()
for label, kw in [("fx", dict(tracer="fx", expand="all")),
                  ('auto, expand="all"', dict(expand="all", example_input=x)),
                  ("export", dict(tracer="export", example_input=x))]:
    try:
        g = ts.trace(fresh(HYBRID), **kw)
    except Exception as e:
        print(f"  {label:20s} {short(e)}")
        continue
    print(f"  {label:20s} {len(g):3d} nodes, source={g.source}, single-tensor cuts {len(g.valid_cuts())}, "
          f"multi-tensor gaps {len(g.multi_tensor_cuts())}")
    for path, row in by_module(g, ["emb", "mid.0", "mid.1", "enc", "head"]).items():
        print(f"      {path:6s} {row}")

print("\nTraining, n=4, multi_tensor=True, and a named cut inside mid.1 (between its l1 and l2):")
ref = train_unsplit(fresh(HYBRID), TOK_DATA)
for label, make in [('auto, expand="all", n=4', lambda m: ts.slice(m, n=4, multi_tensor=True, expand="all", example_input=IDS)),
                    ("export, n=4", lambda m: ts.slice(m, n=4, multi_tensor=True, tracer="export", example_input=IDS)),
                    ('auto, cuts=["mid.0", "mid.1.l1"]', lambda m: ts.slice(m, cuts=["mid.0", "mid.1.l1"], multi_tensor=True,
                                                                     expand="all", example_input=IDS))]:
    m = fresh(HYBRID)
    s = make(m)
    train_sliced(s, TOK_DATA)
    print(f"  {label:32s} cuts send {kib(cut_bytes(s, IDS))}, max |w - unsplit| = {max_diff(m, ref):.1e}")

print("\nSizes the exported blocks assumed are checked: Windowed was exported at T=16, so T > 8.")
g = ts.trace(fresh(HYBRID), expand="all", example_input=IDS)
for T in (12, 5):
    try:
        g.run_nodes(torch.randint(0, VOCAB, (4, T)))
        print(f"  T={T:2d}: runs")
    except ValueError as e:
        print(f"  T={T:2d}: {short(e)}")

section("5b. Alternating as deep as needed: fx -> export -> fx -> export")


@torch.fx.wrap
def gate(h):
    """Reads a value: export can't trace it; fx keeps it as one call."""
    return h * (1.0 if float(h.abs().amax()) > 0 else 0.0)


class Gated(nn.Module):              # fx can trace it (gate is a leaf for fx); export can't
    def __init__(self, d=32):
        super().__init__()
        self.inner, self.lin = Windowed(d), nn.Linear(d, d)

    def forward(self, h):
        return gate(self.lin(self.inner(h)))


class Outer(nn.Module):              # fx can't (a branch on a size): exported, Gated inside
    def __init__(self, d=32):
        super().__init__()
        self.pre, self.gated, self.post = nn.Linear(d, d), Gated(d), nn.Linear(d, d)

    def forward(self, h):
        if h.shape[1] > 4:
            h = torch.relu(self.pre(h))
        return self.post(self.gated(h))


class Deep(nn.Module):
    def __init__(self, vocab=VOCAB, d=32):
        super().__init__()
        self.emb, self.outer, self.head = nn.Embedding(vocab, d), Outer(d), nn.Linear(d, vocab)

    def forward(self, ids):
        return self.head(self.outer(self.emb(ids))).transpose(1, 2)


DEEP = Deep()
g = ts.trace(fresh(DEEP), expand="all", example_input=IDS)
for path in ["emb", "outer", "outer.gated", "outer.gated.inner", "head"]:
    inside = [n for n in g.nodes if (path in n.scope or n.name == path)
              and not any(c in n.scope for c in ["outer.gated", "outer.gated.inner"] if c != path and c.startswith(path))]
    kinds = sorted({origin(n) for n in inside if type(n.module).__name__ != "_AttrWrapper"})
    print(f"  {path:18s} own nodes: {len(inside):3d}, made by {', '.join(kinds)}")
ref = train_unsplit(fresh(DEEP), TOK_DATA)
m = fresh(DEEP)
train_sliced(ts.slice(m, n=4, multi_tensor=True, expand="all", example_input=IDS), TOK_DATA)
print(f"  n=4, multi_tensor: max |w - unsplit| = {max_diff(m, ref):.1e}")
