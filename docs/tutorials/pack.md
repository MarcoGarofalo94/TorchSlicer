# When tracing fails: slicing any model with `pack=`

By default `ts.slice` traces your model's `forward` with `torch.fx` to find its layers and the
places where it can be cut. Some models can't be traced, or can't be represented as a graph
with one input and one output. Then `ts.slice` raises `ts.TraceError` and stops. It does
not guess. (TorchSlicer ≤ 0.2 quietly fell back to treating the children as a chain, which
gave the wrong model.)

```text
torchslicer.core.model_graph.TraceError: torch.fx could not trace GPT2LMHeadModel.forward: ...

TorchSlicer could not build the layer graph of this model automatically. Options:
  1. Pass pack=: a function model -> list[nn.Module] whose stages each take and return one
     tensor, e.g. ts.slice(model, n=4, pack=my_pack). ...
```

Typical causes:

- **Data-dependent control flow** in the top-level `forward` (`if x.sum() > 0:`), or
  Python-level argument handling, as in most Hugging Face models.
- **Several inputs** (`forward(x, mask)`) or **several outputs** (a tuple, or a `ModelOutput`).
- An in-place op whose result is never used (`h.mul_(2)`).

## First, cheaper fixes

- Pass `example_input=` (one batch): when fx fails, `ts.slice` traces with `torch.export`,
  which handles most Hugging Face models without a pack function ([export.md](export.md)).
- Only the top-level `forward` is traced. Submodules are opaque leaves unless you list them in
  `expand=`. Moving control flow into a submodule is often enough.
- If the graph traces but the cut you want isn't allowed (a skip connection crosses it), cut
  somewhere else, or pass `multi_tensor=True` to send every tensor that crosses it
  ([multi_tensor.md](multi_tensor.md)). `ts.trace(model).summary()` marks the valid cuts. See
  [custom_strategy.md](custom_strategy.md).

## The `pack` contract

```python
sliced = ts.slice(model, n=4, pack=my_pack)
```

`pack(model)` returns a list of `nn.Module` *stages*:

- Stage 0 receives the batch input. Stage *i* receives stage *i−1*'s output. Each stage
  returns **one tensor**.
- The last stage's output goes to the criterion together with the labels.
- Stages are atomic. Strategies cut only *between* stages, and fault-tolerance recovery
  re-slices the same stages. Anything that must not be split, such as a residual block or a
  block plus its rotary embedding, belongs inside one stage.
- **Wrap the model's own submodules; don't copy them.** The stages then share parameters
  with `model`, so tied weights (embedding ↔ LM head) are detected and trained correctly.
- Every stage class must be importable on the workers under the same module path, because
  slices travel with `torch.save`. Library stages and `nn.Sequential` always are. Put your own
  stage classes in a package installed on every worker, or on `PYTHONPATH`.

## Example: a model `torch.fx` can't trace

```python
import torch
import torch.nn as nn
import torchslicer as ts

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm, self.ff = nn.LayerNorm(d), nn.Linear(d, d)

    def forward(self, x, mask=None):
        h = self.ff(self.norm(x))
        return x + (h * mask if mask is not None else h)

class Net(nn.Module):
    def __init__(self, vocab=100, d=32, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(n_layers))
        self.head = nn.Linear(d, vocab)

    def forward(self, ids, mask=None):
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(x[:, -1])           # predict the next token

model = Net()
try:
    ts.slice(model, n=2)
except ts.TraceError as e:
    print(e.args[0].splitlines()[0])   # "...uses a second input 'mask'..."
```

Each block already takes and returns one tensor when `mask` is omitted. The head needs the
last position, so a small `nn.Module` does that step:

```python
class LastToken(nn.Module):        # defined in an importable module in real use
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, x):
        return self.head(x[:, -1])

def pack_net(model):
    return [ts.SimpleEmbedStage(model.embed),
            *[ts.BlockStage(b) for b in model.blocks],
            LastToken(model.head)]

sliced = ts.slice(model, n=3, pack=pack_net)
print(sliced.summary())
```

**Check the pack before training.** Chained in order, the stages must compute exactly what
`model` computes:

```python
ids = torch.randint(0, 100, (4, 16))
with torch.no_grad():
    assert torch.allclose(nn.Sequential(*pack_net(model))(ids), model(ids), atol=1e-6)
```

## Hugging Face language models

Ready-made pack functions are in `examples/pack/`:

| File | Models |
|---|---|
| `gpt2.py` | GPT-2 family |
| `llama.py` | Llama 2/3, Mistral, Qwen2/2.5, DeepSeek dense |
| `bert.py` | BERT encoders |
| `deepseek_moe.py` | Mixture-of-Experts (DeepSeek, Mixtral) |

They are built from the library's stage primitives:

| Stage | Wraps |
|---|---|
| `ts.SimpleEmbedStage(embed)` | a token embedding: `int64 [B, T]` → `[B, T, H]` |
| `ts.GPT2EmbedStage(wte, wpe, drop)` | GPT-2 token + position embedding |
| `ts.BlockStage(block, rotary_emb=None)` | one transformer block. It unwraps tuple outputs, and computes RoPE `position_embeddings` when given `rotary_emb` |
| `ts.MoEBlockStage(block, aux_loss_weight, rotary_emb)` | an MoE block returning `(hidden, aux_loss)`. The router loss is backpropagated on the worker that holds the block |
| `ts.CausalLMHeadStage(norm, lm_head)` | final norm + LM head |
| `ts.AuxInputStage` (subclass it) | a first stage that also needs named extra inputs, such as `pixel_values` |

```python
from transformers import GPT2LMHeadModel
from examples.pack.gpt2 import pack_gpt2

model = GPT2LMHeadModel.from_pretrained("gpt2")
sliced = ts.slice(model, n=4, pack=pack_gpt2)                     # uniform over stages
sliced = ts.slice(model, strategy="params", n=4, pack=pack_gpt2)  # or balance parameters
```

With LoRA, call `ts.peft_unwrap(model)` before packing.

**Extra inputs.** If the DataLoader yields a dict, `"input_ids"` (or the first key) is the main
tensor and the other keys are passed by name to stage 0. Stage 0 must be an `AuxInputStage`.
Every later stage gets a single tensor. Without `pack=`, a traced model takes several inputs
anywhere, a mask read by every layer included: pass one batch as `example_input=`
([export.md](export.md#models-with-several-inputs)).

## What still works with `pack=`

- Every strategy works: `uniform`, `params`, `flops` and `profiled` (with `example_input=`),
  and custom ones. Named cuts use stage indices: `cuts=[2, 5]` cuts after stages 2 and 5.
- `expand=` is rejected, because packed stages are not traced.
- Restoring a checkpoint: pass the same `pack` to `ts.restore_model_from_run(model, run_dir,
  pack=pack_gpt2)`.
