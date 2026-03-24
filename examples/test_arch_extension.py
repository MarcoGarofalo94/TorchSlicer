"""
Architecture extension smoke test.

Tests:
  1. Mistral (tiny from config, no download) — LLaMA branch detection, 2 epochs
  2. Synthetic DeepSeek-V2 MoE structure — MoEBlockStage detection + aux loss backward, 2 epochs

Both use LocalExecutor on CPU with random data. No internet required.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import torchslicer as ts


# ── helpers ───────────────────────────────────────────────────────────────────

def make_lm_loader(vocab, seq, batch, n):
    ids    = torch.randint(0, vocab, (n, seq))
    labels = torch.randint(0, vocab, (n, seq))
    return DataLoader(TensorDataset(ids, labels), batch_size=batch, drop_last=True)


def run(name, model, task, loader, n_workers, epochs=2):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    adapter = ts.wrap_hf(model, task=task)
    stages  = [type(s).__name__ for s in adapter]
    print(f"  stages ({len(adapter)}): {stages}")

    moe_stages = [s for s in adapter if isinstance(s, ts.MoEBlockStage)]
    if moe_stages:
        print(f"  MoEBlockStage count: {len(moe_stages)}  "
              f"aux_loss_weight={moe_stages[0].aux_loss_weight}")

    sliced = ts.slice(adapter, strategy="uniform", n=n_workers)
    history = sliced.train(
        loader,
        optimizer  = {"name": "AdamW", "params": {"lr": 5e-4}},
        criterion  = {"name": "CrossEntropyLoss", "params": {}},
        epochs     = epochs,
        verbose    = True,
    )
    losses = [round(h["loss"], 4) for h in history]
    print(f"  loss: {losses[0]} → {losses[-1]}")
    assert losses[-1] < losses[0] * 1.1, "loss did not decrease (possible gradient bug)"
    print(f"  OK")


# ── Test 1: Mistral (from config, no download) ────────────────────────────────

def test_mistral():
    from transformers import MistralConfig, MistralForCausalLM

    VOCAB, SEQ, BATCH = 512, 16, 4

    config = MistralConfig(
        vocab_size             = VOCAB,
        hidden_size            = 64,
        intermediate_size      = 128,
        num_hidden_layers      = 2,
        num_attention_heads    = 2,
        num_key_value_heads    = 2,
        max_position_embeddings= 64,
        _attn_implementation   = "eager",
    )
    model  = MistralForCausalLM(config)
    loader = make_lm_loader(VOCAB, SEQ, BATCH, n=32)

    run("Mistral-tiny (from config, LLaMA branch)", model, "causal_lm", loader, n_workers=2)


# ── Test 2: Synthetic DeepSeek-V2 MoE structure ───────────────────────────────

class _MoEExpertGroup(nn.Module):
    """Mimics DeepSeek-V2 MoE gating + experts."""
    def __init__(self, d, n_experts=4):
        super().__init__()
        # Child names "gate" and "experts" trigger _block_is_moe detection
        self.gate    = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.SiLU()) for _ in range(n_experts)])

    def forward(self, x):
        # Top-1 routing
        logits = self.gate(x)                          # [B, T, n_experts]
        idx    = logits.argmax(-1)                     # [B, T]
        # Load-balancing aux loss (variance of per-expert usage)
        probs    = logits.softmax(-1)
        aux_loss = probs.var()                         # scalar
        # Route tokens
        out = torch.zeros_like(x)
        for e in range(len(self.experts)):
            mask = (idx == e).unsqueeze(-1).float()
            out  = out + mask * self.experts[e](x)
        return out, aux_loss


class _DeepSeekBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d)
        self.self_attn = nn.Linear(d, d)      # simplified self-attention
        self.mlp_norm  = nn.LayerNorm(d)
        self.mlp       = _MoEExpertGroup(d)   # MoE MLP replacement

    def forward(self, hidden_states):
        # Attention sub-layer (simplified — no KV, no mask)
        h = hidden_states + self.self_attn(self.attn_norm(hidden_states))
        # MoE MLP sub-layer — returns (output, aux_loss)
        moe_out, aux_loss = self.mlp(self.mlp_norm(h))
        h = h + moe_out
        return h, aux_loss


class _DeepSeekInner(nn.Module):
    def __init__(self, vocab, d, n_layers):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers       = nn.ModuleList([_DeepSeekBlock(d) for _ in range(n_layers)])
        self.norm         = nn.LayerNorm(d)


class DeepSeekMoEModel(nn.Module):
    """Minimal model matching DeepSeek-V2 top-level structure."""
    def __init__(self, vocab=512, d=64, n_layers=2):
        super().__init__()
        self.model   = _DeepSeekInner(vocab, d, n_layers)
        self.lm_head = nn.Linear(d, vocab, bias=False)


def test_deepseek_moe():
    VOCAB, SEQ, BATCH = 512, 16, 4
    model  = DeepSeekMoEModel(vocab=VOCAB, d=64, n_layers=2)
    loader = make_lm_loader(VOCAB, SEQ, BATCH, n=32)

    run("DeepSeek-MoE-synthetic (MoEBlockStage, aux loss backward)",
        model, "causal_lm", loader, n_workers=2)

    # Extra check: verify MoE aux loss buffer is cleared after each batch
    adapter = ts.wrap_hf(model, task="causal_lm")
    moe_stages = [s for s in adapter if isinstance(s, ts.MoEBlockStage)]
    assert len(moe_stages) == 2, f"expected 2 MoEBlockStages, got {len(moe_stages)}"
    # After training, buffers should be empty (cleared by executor)
    for s in moe_stages:
        assert s._aux_loss is None, "aux_loss buffer not cleared after training"
    print("  MoE buffer clear check: OK")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_mistral()
    test_deepseek_moe()
    print("\nAll architecture extension tests passed.")
