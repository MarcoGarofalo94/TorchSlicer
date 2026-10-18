"""Inverting a language model's internal states at the cut back to the prompt tokens.

Embedding Recovery (ER), the white-box attack of Dong et al., "Depth Gives a False Sense of
Privacy: LLM Internal States Inversion", USENIX Security 2025,
https://arxiv.org/abs/2507.16372 (§4.2, Algorithm 1). The server knows the client layers'
weights (e.g. a public pretrained model whose early layers the client keeps frozen) and
receives their hidden states ``h``. It optimizes dummy input embeddings ``w^`` (zero
initialized) so that the client layers map them to ``h``, keeps the iterate of lowest
loss, and reads each position's token as the vocabulary embedding of highest cosine
similarity with it. The paper's defaults: AdamW, MSE distance, lr 5e-4, no distribution
penalty (lambda = 0), 5e4 steps; fewer steps here by default.
"""

from collections import Counter

import torch
import torch.nn as nn

from .base import Attack, register_attack


def token_f1(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Mean over sequences of the bag-of-tokens F1 between predicted and true ids."""
    scores = []
    for p, t in zip(pred.tolist(), true.tolist()):
        common = sum((Counter(p) & Counter(t)).values())
        scores.append(0.0 if common == 0 else 2 * common / (len(p) + len(t)))
    return sum(scores) / len(scores)


def _embedding(module: nn.Module, name: str | None) -> nn.Embedding:
    if name is not None:
        return module.get_submodule(name)
    embs = [m for m in module.modules() if isinstance(m, nn.Embedding)]
    if not embs:
        raise ValueError("EmbeddingRecovery found no nn.Embedding in the client slices; pass embedding=")
    return max(embs, key=lambda m: m.num_embeddings)        # the token table, not positions


class EmbeddingRecovery(Attack):
    """Recover the token ids the client slices embedded from the hidden states at ``cut``.

    On every ``every``-th observed batch it inverts the first ``n_invert`` sequences:
    ``steps`` AdamW steps (``lr``) on the dummy embeddings, distance ``"mse"`` or
    ``"cos"``. ``embedding``: the token embedding's module name inside the client copy
    (``slices.0.layers.0.wte``-style); default the largest ``nn.Embedding``, whose output
    is replaced by the dummy embeddings. Inputs must be token ids ``(N, T)`` and the
    hidden states ``(N, T, H)``. Metrics: ``token_acc`` (same token at the same position)
    and ``token_f1`` (bag of tokens, as in the paper).
    """
    name = "embedding_recovery"
    needs_client = True

    def __init__(self, cut: int = 0, steps: int = 500, lr: float = 5e-4, distance: str = "mse",
                 n_invert: int = 4, every: int = 10, embedding: str | None = None, key: str | None = None):
        super().__init__(cut, key)
        if distance not in ("mse", "cos"):
            raise ValueError("distance must be 'mse' or 'cos'")
        self.steps, self.lr, self.distance = int(steps), float(lr), distance
        self.n_invert, self.every, self.embedding = int(n_invert), max(1, int(every)), embedding
        self._seen = 0
        self.reset()

    def reset(self):
        self.accs, self.f1s = [], []

    def _dist(self, out, h):
        if self.distance == "mse":
            return (out - h).pow(2).mean()
        return (1 - nn.functional.cosine_similarity(out, h, dim=-1)).mean()

    def observe(self, obs):
        self._seen += 1
        if (self._seen - 1) % self.every:
            return
        ids = obs.inputs[:self.n_invert]
        if ids.is_floating_point() or ids.dim() != 2:
            raise ValueError(f"EmbeddingRecovery needs token ids (N, T) as inputs, got {tuple(ids.shape)} "
                             f"{ids.dtype}")
        h = obs.tensor[:self.n_invert].float()
        client = self.client()
        emb = _embedding(client, self.embedding)
        table = emb.weight.detach().float()
        w = torch.zeros(ids.shape + (table.shape[1],), device=h.device, requires_grad=True)
        handle = emb.register_forward_hook(lambda m, i, o: w.to(o.dtype))
        dummy = torch.zeros_like(ids)               # the token ids are what the attacker lacks
        opt = torch.optim.AdamW([w], lr=self.lr)
        best, best_w = float("inf"), w.detach().clone()
        try:
            for _ in range(self.steps):
                loss = self._dist(client(dummy).float(), h)
                if loss.item() < best:
                    best, best_w = loss.item(), w.detach().clone()
                opt.zero_grad()
                loss.backward()
                opt.step()
        finally:
            handle.remove()
        cos = nn.functional.normalize(best_w, dim=-1) @ nn.functional.normalize(table, dim=-1).T
        pred = cos.argmax(-1)
        self.accs.append((pred == ids).float().mean().item())
        self.f1s.append(token_f1(pred.cpu(), ids.cpu()))

    def metrics(self):
        if not self.accs:
            return {}
        return {"token_acc": round(sum(self.accs) / len(self.accs), 4),
                "token_f1": round(sum(self.f1s) / len(self.f1s), 4)}


register_attack("embedding_recovery", EmbeddingRecovery)
