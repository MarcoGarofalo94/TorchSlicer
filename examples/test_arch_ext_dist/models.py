"""
Synthetic DeepSeek-V2 MoE model for the arch extension distributed test.

Imported by both coordinator and worker so torch.load() can deserialize the classes.
"""

import torch
import torch.nn as nn


class _MoEExpertGroup(nn.Module):
    def __init__(self, d, n_experts=4):
        super().__init__()
        self.gate    = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.SiLU())
                                      for _ in range(n_experts)])

    def forward(self, x):
        logits   = self.gate(x)
        idx      = logits.argmax(-1)
        probs    = logits.softmax(-1)
        aux_loss = probs.var()
        out = torch.zeros_like(x)
        for e in range(len(self.experts)):
            mask = (idx == e).unsqueeze(-1).float()
            out  = out + mask * self.experts[e](x)
        return out, aux_loss


class _DeepSeekBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d)
        self.self_attn = nn.Linear(d, d)
        self.mlp_norm  = nn.LayerNorm(d)
        self.mlp       = _MoEExpertGroup(d)

    def forward(self, hidden_states):
        h = hidden_states + self.self_attn(self.attn_norm(hidden_states))
        moe_out, aux_loss = self.mlp(self.mlp_norm(h))
        return h + moe_out, aux_loss


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
