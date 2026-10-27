"""Slice real architectures with `multi_tensor=True` (torchvision CNNs and vision transformers,
segmentation models, GPT-style and encoder-decoder transformers) and check one SGD step equals
the unsplit model's; also runs a Quantize cut hook on every cut. CPU, a few seconds per model.

    python benchmarks/multi_tensor/model_sweep.py [model ...]
"""
import copy, sys, time, warnings, traceback
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as tvm
import torchslicer as ts
from torchslicer.core.multicut import packed_cuts
warnings.filterwarnings("ignore")
torch.set_num_threads(4)

class MiniGPT(nn.Module):           # hand-written attention, causal mask + positions in forward, tied head
    def __init__(s, V=100, d=64, L=4, h=4):
        super().__init__()
        s.emb, s.pos = nn.Embedding(V, d), nn.Embedding(128, d)
        s.blocks = nn.ModuleList(nn.ModuleDict(dict(ln1=nn.LayerNorm(d), qkv=nn.Linear(d, 3*d), o=nn.Linear(d, d),
                                                    ln2=nn.LayerNorm(d), ff=nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))))
                                 for _ in range(L))
        s.ln, s.head, s.h = nn.LayerNorm(d), nn.Linear(d, V, bias=False), h
        s.head.weight = s.emb.weight
    def forward(s, ids):
        B, T = ids.shape
        mask = ids.new_ones(T, T).tril().bool()          # torch.ones(T, T) with traced T: fx cannot
        x = s.emb(ids) + s.pos(torch.arange(T, device=ids.device))
        for b in s.blocks:
            q, k, v = b["qkv"](b["ln1"](x)).chunk(3, -1)
            q, k, v = (t.view(B, T, s.h, -1).transpose(1, 2) for t in (q, k, v))
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            x = x + b["o"](a.transpose(1, 2).reshape(B, T, -1))
            x = x + b["ff"](b["ln2"](x))
        return s.head(s.ln(x)).transpose(1, 2)

class TorchEncDec(nn.Module):       # nn.Transformer: encoder output read by every decoder layer
    def __init__(s, V=100, d=64):
        super().__init__()
        s.emb = nn.Embedding(V, d)
        s.tf = nn.Transformer(d, 4, 2, 2, 128, dropout=0.0, batch_first=True)
        s.out = nn.Linear(d, V)
    def forward(s, ids):
        T = ids.size(1)
        src = s.emb(ids)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=ids.device)
        return s.out(s.tf(src, src, tgt_mask=mask)).transpose(1, 2)

class TorchGPT(nn.Module):          # nn.TransformerEncoder with a causal mask built in forward
    def __init__(s, V=100, d=64):
        super().__init__()
        s.emb, s.pos = nn.Embedding(V, d), nn.Parameter(torch.zeros(1, 64, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 128, dropout=0.1, batch_first=True), 3,
                                      enable_nested_tensor=False)
        s.out = nn.Linear(d, V)
    def forward(s, ids):
        T = ids.size(1)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), 1)
        return s.out(s.enc(s.emb(ids) + s.pos[:, :T], mask=mask)).transpose(1, 2)

class Seg(nn.Module):               # torchvision segmentation: dict output -> ["out"]
    def __init__(s, make):
        super().__init__(); s.m = make()
    def forward(s, x):
        return s.m(x)["out"]

img = lambda n=1000, hw=64: (lambda: torch.randn(4, 3, hw, hw), lambda: torch.randint(0, n, (4,)))
tok = (lambda: torch.randint(0, 100, (4, 16)), lambda: torch.randint(0, 100, (4, 16)))
seg = (lambda: torch.randn(2, 3, 64, 64), lambda: torch.randint(0, 5, (2, 64, 64)))
MODELS = {
    "resnet18": (lambda: tvm.resnet18(), *img()),
    "resnet50": (lambda: tvm.resnet50(), *img()),
    "mobilenet_v3_small": (lambda: tvm.mobilenet_v3_small(), *img()),
    "efficientnet_b0": (lambda: tvm.efficientnet_b0(), *img()),
    "densenet121": (lambda: tvm.densenet121(), *img()),
    "convnext_tiny": (lambda: tvm.convnext_tiny(), *img()),
    "regnet_y_400mf": (lambda: tvm.regnet_y_400mf(), *img()),
    "shufflenet_v2": (lambda: tvm.shufflenet_v2_x0_5(), *img()),
    "squeezenet": (lambda: tvm.squeezenet1_0(), *img()),
    "googlenet": (lambda: tvm.googlenet(aux_logits=False, init_weights=True), *img()),
    "vit_b_16": (lambda: tvm.vit_b_16(image_size=64), *img()),
    "swin_t": (lambda: tvm.swin_t(), *img()),
    "maxvit_t": (lambda: tvm.maxvit_t(), *img(hw=224)),
    "fcn_resnet50": (lambda: Seg(lambda: tvm.segmentation.fcn_resnet50(weights=None, weights_backbone=None, num_classes=5, aux_loss=False)), *seg),
    "lraspp_mobilenet": (lambda: Seg(lambda: tvm.segmentation.lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=5)), *seg),
    "minigpt": (MiniGPT, *tok),
    "torch_encdec": (TorchEncDec, *tok),
    "torch_gpt": (TorchGPT, *tok),
}

def step(model, run, x, y):
    torch.manual_seed(1)
    return run(model, x, y)

def unsplit(model, x, y):
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()

def main(names):
    for name in names:
        make, xf, yf = MODELS[name]
        t0 = time.time()
        try:
            torch.manual_seed(0); base = make().train()
            torch.manual_seed(2); x, y = xf(), yf()
            g = ts.trace(base, expand="all")
            single = len(g.valid_cuts())
            ref = copy.deepcopy(base); step(ref, unsplit, x, y)
            row = f"{name:20s} nodes={len(g):4d} single-tensor cuts={single:4d}"
            for n in (2, 4):
                m = copy.deepcopy(base)
                s = ts.slice(m, n=n, expand="all", multi_tensor=True)
                pk = len(packed_cuts(s.graph, s.partitions))
                torch.manual_seed(1)
                h = s.train([(x, y)], {"name": "SGD", "params": {"lr": 0.01}}, {"name": "CrossEntropyLoss"}, epochs=1, verbose=False)
                d = max((a - b).abs().max().item() for a, b in zip(m.state_dict().values(), ref.state_dict().values()) if a.is_floating_point())
                kb = [round(b / 1024) for b in h[0]["bytes_fwd"]]
                row += f" | n={n}: packed {pk}/{n-1}, KiB/cut {kb}, max|d|={d:.1e}"
            # a lossy hook on every cut, packed or not, runs
            m = copy.deepcopy(base)
            ts.slice(m, n=4, expand="all", multi_tensor=True, cut_hooks=ts.Quantize(bits=8)).train(
                [(x, y)], {"name": "SGD", "params": {"lr": 0.01}}, {"name": "CrossEntropyLoss"}, epochs=1, verbose=False)
            row += " | quantize hook OK"
            print("OK  ", row, f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print("FAIL", f"{name:20s}", f"{type(e).__name__}: {str(e).splitlines()[0][:220]}", flush=True)

if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
