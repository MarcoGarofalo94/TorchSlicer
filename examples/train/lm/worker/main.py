"""
TinyGPT P2P worker — character-level language model via split learning.

Same P2P driver/follower topology as the ResNet18 example, with a small
causal transformer instead.  The driver owns the corpus and DataLoader;
the model is sliced into N partitions and distributed across workers.

Model  : TinyGPT — 4 transformer blocks, d_model=128, 4 heads  (~830K params)
Data   : byte-level LM on all .md files under /workspace (corpus mounted at runtime)
Task   : predict next byte from a 64-byte context window
Input  : int64 token IDs  (SplitLayer already handles non-float inputs correctly)

The P2P driver/follower logic is the library's (``ts.run_p2p``); this file only
defines the model and the data.

Environment variables:
  IS_DRIVER          true/false (default false)
  WORKER_PEERS       comma-separated "host:port" list in slice order, driver first
  WORKER_ADDRESS     address advertised to peers
  EXPERIMENT_CONFIG  path to YAML experiment config
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import torchslicer as ts


# ── model ─────────────────────────────────────────────────────────────────────

VOCAB_SIZE = 256    # byte-level
D_MODEL    = 128
N_HEADS    = 4
N_LAYERS   = 4
MAX_SEQ    = 64


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.d_head  = D_MODEL // N_HEADS
        self.qkv     = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.proj    = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(MAX_SEQ, MAX_SEQ)).view(1, 1, MAX_SEQ, MAX_SEQ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL, bias=False),
            nn.GELU(),
            nn.Linear(4 * D_MODEL, D_MODEL, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln2  = nn.LayerNorm(D_MODEL)
        self.ff   = FeedForward()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TokenEmbedding(nn.Module):
    """Token + positional embedding. Accepts int64 byte IDs."""

    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(MAX_SEQ, D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return self.tok(x) + self.pos(torch.arange(T, device=x.device))


class LMHead(nn.Module):
    """LayerNorm + linear projection. Returns logits for the last position only."""

    def __init__(self):
        super().__init__()
        self.ln   = nn.LayerNorm(D_MODEL)
        self.proj = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.ln(x[:, -1, :]))   # [B, VOCAB_SIZE]


class TinyGPT(nn.Module):
    """
    Flat sequential model: each direct child is an opaque leaf for TorchSlicer's
    shallow tracer.  Input: int64 byte IDs [B, MAX_SEQ].  Output: [B, VOCAB_SIZE].
    """

    def __init__(self):
        super().__init__()
        self.embed   = TokenEmbedding()
        self.block_0 = TransformerBlock()
        self.block_1 = TransformerBlock()
        self.block_2 = TransformerBlock()
        self.block_3 = TransformerBlock()
        self.head    = LMHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        return self.head(x)


def build_model() -> nn.Module:
    return TinyGPT()


# ── dataset ───────────────────────────────────────────────────────────────────

def get_dataset(data_dir: str = "/workspace", batch_size: int = 64,
                n_train: int = 50_000):
    """Byte-level LM on .md files under data_dir. Samples up to n_train examples."""
    corpus = bytearray()
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".md"):
            path = os.path.join(data_dir, fname)
            try:
                corpus += open(path, "rb").read()
            except OSError:
                pass

    if len(corpus) < MAX_SEQ + 2:
        raise RuntimeError(
            f"corpus too short ({len(corpus)} bytes); need at least {MAX_SEQ + 2}")

    ids = torch.frombuffer(bytes(corpus), dtype=torch.uint8).long()
    n   = min(len(ids) - MAX_SEQ, n_train)

    # random subsample for reproducibility
    torch.manual_seed(42)
    indices = torch.randperm(len(ids) - MAX_SEQ)[:n]
    inputs  = torch.stack([ids[i : i + MAX_SEQ] for i in indices])
    targets = ids[indices + MAX_SEQ]

    print(f"[dataset] corpus={len(corpus):,} bytes  samples={n:,}  "
          f"vocab={VOCAB_SIZE}  seq_len={MAX_SEQ}")
    return DataLoader(TensorDataset(inputs, targets),
                      batch_size=batch_size, shuffle=True, drop_last=True)


# ── entry point ───────────────────────────────────────────────────────────────

def serve() -> list:
    """Run this node (role from IS_DRIVER); the driver returns the history."""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else None
    return ts.run_p2p(build_model, get_dataset, port=port)


if __name__ == "__main__":
    serve()
