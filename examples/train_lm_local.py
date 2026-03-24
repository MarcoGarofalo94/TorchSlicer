#!/usr/bin/env python3
"""
TinyGPT — character-level language model via split learning.

A small causal transformer is sliced into N partitions and trained locally
with TorchSlicer's LocalExecutor.  Demonstrates that TorchSlicer works out
of the box with transformer architectures (int64 token inputs, attention,
residual connections).

Model  : 4-layer causal transformer, d_model=128, 4 heads  (~830K params)
Data   : character-level LM on CLAUDE.md (always available in the repo)
Task   : predict next character from a 64-token context window
Loss   : CrossEntropyLoss on last-position logit vs. true next char

Usage:
    conda run -n torchslicer python3 examples/train_lm_local.py
    conda run -n torchslicer python3 examples/train_lm_local.py --n 2 --epochs 10
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import torchslicer as ts
from torchslicer.config import RunConfig


# ── hyper-parameters ──────────────────────────────────────────────────────────

D_MODEL    = 128
N_HEADS    = 4
N_LAYERS   = 4      # number of transformer blocks
MAX_SEQ    = 64     # context window
DROPOUT    = 0.0    # disabled for deterministic quick runs

BATCH_SIZE = 64
LR         = 3e-4


# ── model ─────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj    = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq, max_seq)).view(1, 1, max_seq, max_seq),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scale = self.d_head ** -0.5
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (self-attention + FFN, residual connections)."""

    def __init__(self, d_model: int, n_heads: int, max_seq: int):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TokenEmbedding(nn.Module):
    """Token + positional embedding, accepts int64 token IDs."""

    def __init__(self, vocab_size: int, d_model: int, max_seq: int):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        return self.tok(x) + self.pos(pos)


class LMHead(nn.Module):
    """Final layer-norm + projection.  Returns logits for the last position."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.ln   = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.ln(x[:, -1, :]))  # [B, vocab_size]


class TinyGPT(nn.Module):
    """
    Flat sequential structure so TorchSlicer's shallow tracer sees each child
    as an opaque leaf: TokenEmbedding → block_0..3 → LMHead.

    int64 token IDs in → [batch, vocab_size] logits out.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.embed   = TokenEmbedding(vocab_size, D_MODEL, MAX_SEQ)
        self.block_0 = TransformerBlock(D_MODEL, N_HEADS, MAX_SEQ)
        self.block_1 = TransformerBlock(D_MODEL, N_HEADS, MAX_SEQ)
        self.block_2 = TransformerBlock(D_MODEL, N_HEADS, MAX_SEQ)
        self.block_3 = TransformerBlock(D_MODEL, N_HEADS, MAX_SEQ)
        self.head    = LMHead(D_MODEL, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        return self.head(x)


# ── dataset ───────────────────────────────────────────────────────────────────

def load_corpus() -> str:
    """Load text from CLAUDE.md and any other .md files in the repo root."""
    root = os.path.join(os.path.dirname(__file__), "..")
    text = ""
    for fname in sorted(os.listdir(root)):
        if fname.endswith(".md"):
            path = os.path.join(root, fname)
            try:
                text += open(path, encoding="utf-8", errors="ignore").read() + "\n"
            except OSError:
                pass
    return text


def build_dataset(text: str, seq_len: int):
    """
    Character-level tokenisation.  Returns:
        loader  : DataLoader of (input_ids [B, seq_len], target_id [B])
        vocab   : sorted list of unique chars
    """
    vocab   = sorted(set(text))
    c2i     = {c: i for i, c in enumerate(vocab)}
    ids     = torch.tensor([c2i[c] for c in text], dtype=torch.long)

    # sliding window: input = ids[i:i+seq_len], target = ids[i+seq_len]
    n       = len(ids) - seq_len
    inputs  = torch.stack([ids[i : i + seq_len] for i in range(n)])
    targets = ids[seq_len:]

    loader = DataLoader(
        TensorDataset(inputs, targets),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    return loader, vocab


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=2,  help="number of partitions")
    parser.add_argument("--epochs", type=int, default=5,  help="training epochs")
    parser.add_argument("--gpipe",  action="store_true",  help="enable GPipe micro-batching")
    args = parser.parse_args()

    # ── data ──────────────────────────────────────────────────────────────────
    print("loading corpus ...")
    text = load_corpus()
    if not text:
        raise RuntimeError("no .md files found — run from the repo root")
    loader, vocab = build_dataset(text, MAX_SEQ)
    vocab_size = len(vocab)
    n_params   = sum(p.numel() for p in TinyGPT(vocab_size).parameters())
    print(f"corpus : {len(text):,} chars  |  vocab : {vocab_size}  |  "
          f"batches/epoch : {len(loader)}")
    print(f"model  : TinyGPT  d_model={D_MODEL}  heads={N_HEADS}  "
          f"blocks={N_LAYERS}  params={n_params:,}")
    print(f"split  : {args.n} partitions  gpipe={args.gpipe}")
    print()

    # ── model & config ────────────────────────────────────────────────────────
    model = TinyGPT(vocab_size)

    cfg = RunConfig()
    cfg.training.epochs    = args.epochs
    cfg.training.optimizer = {"name": "AdamW", "params": {"lr": LR}}
    cfg.training.criterion = {"name": "CrossEntropyLoss", "params": {}}
    cfg.pipeline.use_gpipe = args.gpipe
    cfg.pipeline.n_micro   = 4
    cfg.logging.enabled    = True
    cfg.logging.dir        = "runs"
    cfg.run_id             = f"tinygpt_{args.n}p"

    # ── slice & train ─────────────────────────────────────────────────────────
    sliced = ts.slice(model, strategy="uniform", n=args.n)

    t0 = time.perf_counter()
    sliced.train(
        loader,
        cfg.training.optimizer,
        cfg.training.criterion,
        epochs        = args.epochs,
        verbose       = True,
        use_gpipe     = args.gpipe,
        n_micro_batches = cfg.pipeline.n_micro,
        run_config    = cfg,
    )
    elapsed = time.perf_counter() - t0

    run_dir = os.path.join(cfg.logging.dir, cfg.run_id)
    print()
    print(f"done in {elapsed:.1f}s")
    print(f"logs → {run_dir}/metrics.jsonl")
    print(f"       tail -f {run_dir}/metrics.jsonl")


if __name__ == "__main__":
    main()
