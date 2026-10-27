# Models `torch.fx` can't trace: `tracer="export"`

By default `ts.slice` builds the model's graph with `torch.fx`, which reads `forward`
symbolically, without running it. That fails on many real models, Hugging Face transformers
among them (keyword-heavy signatures, decorators, cache objects, `if` on a shape), and
`ts.slice` raises `ts.TraceError`. `torch.export` runs `forward` instead, on fake tensors
shaped like an example input, so it goes through all of that. Give `ts.slice` an example
batch and it traces with `torch.export`:

```python
import torch
import torch.nn as nn
import torchslicer as ts
from transformers import GPT2Config, GPT2LMHeadModel

class CausalLM(nn.Module):
    """logits as [batch, vocab, length], what nn.CrossEntropyLoss takes."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids):
        return self.model(ids).logits.transpose(1, 2)

model = CausalLM(GPT2LMHeadModel(GPT2Config(n_layer=4, n_embd=64, n_head=4, vocab_size=1000)))
ids = torch.randint(0, 1000, (8, 32))

sliced = ts.slice(model, n=4, multi_tensor=True, tracer="export", example_input=ids)
print(sliced.summary(example_input=ids))
```

```text
SlicedModel(CausalLM, strategy=uniform, 4 slices, 329,600 params)
  slice 0: 75 nodes, 146,304 params
      input, ..., model.transformer.wte (2 nodes), ..., model.transformer.h.0.attn (33 nodes), ...
  ── cut 0 after add_5: 4 values packed (add_5, sym_size_int_18, sym_size_int_17, sym_size_int): [8, 2093], 65.4 KiB ──
  slice 1: 110 nodes, 83,200 params
      arange_3.cut0, ..., expand.cut0, model.transformer.h.0.mlp (19 nodes), add_8, model.transformer.h.1 (60 nodes), ...
  ...
  ── output: [8, 1000, 32], 1000.0 KiB ──
```

Each cut sends the hidden state (`[8, 32, 64]`, 64 KiB) and three sizes; the causal mask
and the positions are rebuilt after the cut (`expand.cut0`, ...), not sent. Training is
exact: the sliced model trains like the unsplit one, dropout included.

No `pack=` function, and no class of yours ends up in a slice: the slices hold PyTorch ops
and TorchSlicer's own modules, so workers need nothing but `torchslicer` and `torch`.

## Choosing the tracer

`tracer=` takes `"fx"`, `"export"` or `"auto"`, in `ts.slice` and `ts.trace`:

- `"auto"` (the default) traces with fx; if fx fails and you passed `example_input=`, it
  traces with export. Models fx handles get the same graph as before.
- `"export"` always uses export. Use it when your model wraps another one as a child, as
  `CausalLM` above: fx doesn't trace into children (unless `expand=`), so it "succeeds" with
  the whole wrapped model as one node.
- `"fx"` never uses export.

## What the graph looks like

One node per ATen op (`addmm`, `layer_norm`, `scaled_dot_product_attention`, ...), like
`expand="all"` under fx but finer. Parameters are read by `_AttrWrapper` nodes placed right
before the op that uses them, and they stay the model's own `nn.Parameter`s (tied weights,
such as GPT-2's embedding and head, are trained once). Each node remembers the modules it
came from, so `cuts=` takes module names:

```python
ts.slice(model, cuts=["model.transformer.h.1"], multi_tensor=True, tracer="export", example_input=ids)
```

A strategy that counts nodes (`uniform`) would otherwise cut inside a layer; on an exported
graph each cut moves to the gap that sends the fewest bytes nearby (within half the distance
to the next cut). Export records every value's shape, so this needs no profiling.

Language models need `multi_tensor=True`: the mask, the positions and the sizes reach every
block, so without it almost no cut has a single tensor crossing
([multi_tensor.md](multi_tensor.md)).

## The example input

- Its batch must be at least 2 (export fixes a size of 1). Sizes that the model doesn't fix
  stay symbolic: other batch sizes and other lengths run on the same slices.
- Sizes the model does fix (the image size of a model with a fixed-size head, say) are
  checked at the first slice, with an error that says what was expected.
- It must be one tensor. The model's output can be a tensor, a dict or `ModelOutput` with
  `"logits"` (that one is used), or anything holding exactly one tensor.
- Tracing doesn't change the model: no weights, buffers or random state.

## Training and evaluation

The graph is the model's training-mode `forward`. Dropout (also inside
`scaled_dot_product_attention`), batch norm and its running statistics follow each slice's
`train()`/`eval()`, so evaluation is right. Any other `if self.training:` branch in your code
is fixed at its training value.

## Restoring a run

`ts.restore_model_from_run(model, run_dir)` exports the model again with the recorded input
shape, so it gets the same graph; nothing to pass.

## When export fails too

Export can't trace control flow on values (`if x.sum() > 0:`), or a model with several
inputs. Then use `pack=` ([pack.md](pack.md)). `ts.TraceError` says which tracer failed and
why.
