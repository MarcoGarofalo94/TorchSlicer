# Cutting through skip connections: `multi_tensor=True`

By default exactly one tensor crosses each cut: the output of the last node before it. That
keeps the split-learning contract simple (one activation forward, one gradient back), but it
rules out every cut that a skip connection spans. Some models then have almost no place to
cut:

- **U-Nets**: every encoder level is read again by the decoder;
- **encoder-decoder transformers** (T5, BART, Whisper): every decoder layer reads the encoder
  output;
- models that **read their input again** late, or that build a **mask or positions** in
  `forward` that every block reads.

`ts.slice(..., multi_tensor=True)` lets you cut anywhere. At a cut that several values cross,
TorchSlicer sends all of them, packed into one tensor, and unpacks them on the other side.

```python
import torch
import torch.nn as nn
import torchslicer as ts

class UNet(nn.Module):
    def __init__(self, c=8):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(3, c, 3, padding=1), nn.ReLU())
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

model = UNet()
print(ts.trace(model).valid_cuts())          # [0, 7]: only right after e1 and before d1

sliced = ts.slice(model, cuts=["e2", "d2"], multi_tensor=True)
print(sliced.summary(example_input=torch.randn(4, 3, 32, 32)))
```

```text
SlicedModel(UNet, strategy=named, 3 slices, 7,379 params)
  slice 0: 3 nodes, 1,392 params
      e1, e2
  ── cut 0 after e2: 2 values packed (e2, e1): [4, 12322], 192.5 KiB ──
  slice 1: 7 nodes, 5,672 params
      mid, up2, cat, d2
  ── cut 1 after d2: 2 values packed (d2, e1): [4, 10274], 160.5 KiB ──
  slice 2: 5 nodes, 315 params
      up1, cat_1, d1
  ── output: [4, 3, 32, 32], 48.0 KiB ──
```

The first cut sends `e2`'s output and `e1`'s (the skip the decoder reads at the end); the
second sends `d2`'s and `e1`'s again, passed on by slice 1. A strategy works the same way:
with `multi_tensor=True` every gap is a candidate, so `ts.slice(model, n=4,
multi_tensor=True)` or `strategy="params"` can cut anywhere.

Training is unchanged and exact: the sliced model trains like the unsplit one.

```python
x, y = torch.randn(8, 3, 32, 32), torch.randint(0, 3, (8, 32, 32))
history = sliced.train([(x, y)], {"name": "SGD", "params": {"lr": 0.05}},
                       {"name": "CrossEntropyLoss"}, epochs=1)
print(history[0]["bytes_fwd"])               # bytes sent forward at each cut
```

## What is sent

A cut that one tensor crosses is left exactly as before. At a cut that several values cross,
the graph gets a `cut<k>.send` node at the end of the slice and one `cut<k>.recv<i>` node per
value at the start of the next one. Executors, workers and transports still move one tensor
per cut, so this works everywhere: local, GPipe, offload, centralized and P2P (gRPC, TCP,
IPC, NCCL), behind NAT, `ClientSession`, client nodes, recovery from a dead worker, and
`ts.restore_model_from_run`.

The packed tensor is `[batch, header + values]` in the widest floating-point dtype among the
values (casting is lossless and undone on the other side). A value whose first dimension is
the batch size is laid out per sample; anything else (a parameter, a `[1, T, d]` table, a
Python number) is repeated in every row and read from the first. Integer and bool tensors
travel as their bytes. A value can also be `None`, a `torch.dtype` or a tuple/list of values
(e.g. `nn.MultiheadAttention` returns `(output, weights)`). Gradients flow back through the
packing, and `bytes_fwd`/`bytes_bwd` count the packed tensor.

## Masks and positions are rebuilt, not sent

Language models typically build a causal mask and position ids in `forward`, from the
input's length. Every block reads the mask, so it crosses every cut between blocks. With
`multi_tensor=True` such a value is **rebuilt** after the cut instead: only what it is built
from crosses (the length, the device), and a copy of its nodes runs in the receiving slice.

```python
class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, 2, batch_first=True)
        self.ff = nn.Linear(d, d)

    def forward(self, h, mask):
        return h + self.ff(self.attn(h, h, h, attn_mask=mask, need_weights=False)[0])

class Decoder(nn.Module):
    def __init__(self, vocab=50, d=16, n=3):
        super().__init__()
        self.emb, self.pos = nn.Embedding(vocab, d), nn.Embedding(64, d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(n))
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        T = ids.size(1)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), 1)
        h = self.emb(ids) + self.pos(torch.arange(T, device=ids.device))
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(h).transpose(1, 2)

sliced = ts.slice(Decoder(), n=3, multi_tensor=True)
print(sliced.summary())                      # slices 1 and 2 hold full.cut0 / triu.cut0, ...
```

A value is rebuilt only when it is computed from shapes (`x.size(1)`), devices, dtypes and
constant buffers, through plain operations (no parameter, no module of yours, which could
keep state). Random operations are never rebuilt: drawing again would change the random
numbers, so their results are sent.

## See it

`sliced.view(example_input=x)` draws the model's own graph with the cuts; each packed cut
says how many values it sends and how many bytes. With `multi_tensor=True` every gap can be
clicked, and the page writes the matching `ts.slice(..., multi_tensor=True)` call.
`ts.trace(model).view(multi_tensor=True)` does the same before slicing.

## Not yet

- Cut hooks (compression, noise, `ClipNoise`) and privacy attacks or defences can't be placed
  on a packed cut: they would see the packed layout. Use them on cuts one tensor crosses;
  `ts.slice` and `train` refuse the others with a clear error.
- Capability-aware planning (strategies taking `nodes=`) and recovery re-plan among the cuts
  of the rewritten graph: the original cut points and the cuts one tensor crosses.
- Under bf16 mixed precision, results across any cut can differ from one slice by rounding:
  a value that was a non-contiguous view arrives as a contiguous tensor, and the GPU may pick
  another matrix-multiply algorithm for it. In float32 they are equal.
