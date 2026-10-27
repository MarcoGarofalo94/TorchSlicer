"""Shared model/data definitions for the TorchSlicer model x mode matrix.

Every builder re-seeds, so the reference, local, coordinator and P2P-driver
processes all start from identical weights and see identical batches.
"""
import json
import os

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, TensorDataset

DEVICE = os.environ.get("MATRIX_DEVICE", "cpu")
N_SAMPLES, BATCH, EPOCHS = 64, 16, 2
N_PARTS = int(os.environ.get("MATRIX_PARTS", 3))   # slices; the unsplit reference doesn't depend on it
VOCAB, SEQ = 100, 16
OPTIMIZER = json.loads(os.environ.get("MATRIX_OPT", '{"name": "SGD", "params": {"lr": 0.05, "momentum": 0.9}}'))
CRITERION = {"name": "CrossEntropyLoss", "params": {}}


def _dnn():
    return nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return self.fc(torch.flatten(x, 1))


def _resnet18():
    m = torchvision.models.resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m


def _tinygpt():
    from examples.train_lm_local import TinyGPT
    return TinyGPT(VOCAB)


def _gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(vocab_size=VOCAB, n_positions=SEQ, n_embd=64, n_layer=4, n_head=4,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    return GPT2LMHeadModel(cfg)


def _llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=SEQ,
                      attention_dropout=0.0)
    return LlamaForCausalLM(cfg)


class UNet(nn.Module):
    """M18: skips at every level (only cuts at the ends are single-tensor)."""

    def __init__(self, c=8, classes=3):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(3, c, 3, padding=1), nn.GroupNorm(2, c), nn.ReLU())
        self.e2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(c, 2 * c, 3, padding=1), nn.ReLU())
        self.mid = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(2 * c, 2 * c, 2, stride=2)
        self.d2 = nn.Sequential(nn.Conv2d(4 * c, c, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(c, c, 2, stride=2)
        self.d1 = nn.Conv2d(2 * c, classes, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(a)
        h = self.d2(torch.cat([self.up2(self.mid(b)), b], 1))
        return self.d1(torch.cat([self.up1(h), a], 1))


class DecLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.ff = nn.Linear(d, d)

    def forward(self, h, mem):
        return h + self.ff(self.attn(h, mem, mem)[0])


class EncDec(nn.Module):
    """M18: every decoder layer cross-attends to the encoder output (a tuple and an int cross)."""

    def __init__(self, d=32, n_dec=4, classes=4):
        super().__init__()
        self.emb = nn.Linear(8, d)
        self.enc = nn.TransformerEncoderLayer(d, 4, 64, dropout=0.0, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 6, d))
        self.dec = nn.ModuleList(DecLayer(d) for _ in range(n_dec))
        self.head = nn.Linear(d, classes)

    def forward(self, x):
        mem = self.enc(self.emb(x))
        h = self.query.expand(x.size(0), -1, -1)
        for layer in self.dec:
            h = layer(h, mem)
        return self.head(h.mean(1))


class MaskBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.ff, self.n = nn.Linear(d, d), nn.LayerNorm(d)

    def forward(self, h, mask):
        return h + self.ff(self.n(h + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]))


class MaskDecoder(nn.Module):
    """M18: positions and a causal mask built in forward; every block reads the mask (rebuilt
    after each cut, not sent)."""

    def __init__(self, vocab=VOCAB, d=32, n=4):
        super().__init__()
        self.emb, self.pos = nn.Embedding(vocab, d), nn.Embedding(SEQ, d)
        self.blocks = nn.ModuleList(MaskBlock(d) for _ in range(n))
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        T = ids.size(1)
        pos = torch.arange(T, device=ids.device)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), 1)
        h = self.emb(ids) + self.pos(pos)
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(h).transpose(1, 2)


MULTI_TENSOR = {"unet", "encdec", "maskdec"}     # cut with ts.slice(..., multi_tensor=True)


def slice_kwargs(name):
    return {"multi_tensor": True} if name in MULTI_TENSOR else {}


LORA_TARGETS = {"tinygpt": ["qkv"], "gpt2": ["c_attn"], "llama": ["q_proj", "v_proj"]}


def _lora(base):
    """LoRA-wrapped base model, unwrapped for ts.slice (frozen base, trainable A/B)."""
    def build():
        from peft import LoraConfig, get_peft_model
        import torchslicer as ts
        model = BUILDERS[base]()
        cfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=LORA_TARGETS[base])
        return ts.peft_unwrap(get_peft_model(model, cfg))
    return build


def _pack(name):
    name = name.removesuffix("-lora")
    if name == "gpt2":
        from examples.pack.gpt2 import pack_gpt2
        return pack_gpt2
    if name == "llama":
        from examples.pack.llama import pack_llama
        return pack_llama
    return None


BUILDERS = {"dnn": _dnn, "cnn": SmallCNN, "resnet18": _resnet18,
            "tinygpt": _tinygpt, "gpt2": _gpt2, "llama": _llama,
            "unet": UNet, "encdec": EncDec, "maskdec": MaskDecoder}
BUILDERS.update({f"{b}-lora": _lora(b) for b in LORA_TARGETS})
MODELS = list(BUILDERS)


def build(name):
    """Return (model, pack_fn) with deterministic init."""
    torch.manual_seed(0)
    return BUILDERS[name](), _pack(name)


def loader(name):
    g = torch.Generator().manual_seed(1)
    if name == "dnn":
        x = torch.randn(N_SAMPLES, 32, generator=g)
        y = torch.randint(0, 4, (N_SAMPLES,), generator=g)
    elif name in ("cnn", "resnet18"):
        x = torch.randn(N_SAMPLES, 3, 32, 32, generator=g)
        y = torch.randint(0, 10, (N_SAMPLES,), generator=g)
    elif name == "unet":
        x = torch.randn(N_SAMPLES, 3, 32, 32, generator=g)
        y = torch.randint(0, 3, (N_SAMPLES, 32, 32), generator=g)
    elif name == "encdec":
        x = torch.randn(N_SAMPLES, 12, 8, generator=g)
        y = torch.randint(0, 4, (N_SAMPLES,), generator=g)
    elif name == "maskdec":
        x = torch.randint(0, VOCAB, (N_SAMPLES, SEQ), generator=g)
        y = torch.roll(x, -1, dims=1)
    else:
        name = name.removesuffix("-lora")
        from examples.train_lm_local import MAX_SEQ
        seq = min(SEQ, MAX_SEQ)
        x = torch.randint(0, VOCAB, (N_SAMPLES, seq), generator=g)
        # tinygpt predicts the next token after the sequence; HF heads emit [B, V, T]
        y = (torch.randint(0, VOCAB, (N_SAMPLES,), generator=g) if name == "tinygpt"
             else torch.roll(x, -1, dims=1))
    return DataLoader(TensorDataset(x, y), batch_size=BATCH, shuffle=False)


def reference_module(name):
    """Unsplit model with the same forward the slices compute (pack stages chained for HF)."""
    model, pack = build(name)
    return nn.Sequential(*pack(model)) if pack else model
