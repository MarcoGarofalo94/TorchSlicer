"""Built-in SL algorithms shipped with TorchSlicer.

All algorithms reproduce the current behaviour by default (DefaultAlgorithm).
Researchers subclass or combine them.
"""

import copy

import torch
import torch.nn as nn
import torch.optim as optim

from .base import Algorithm, AlgorithmContext, BackwardPolicy
from .registry import register


# ── helpers ────────────────────────────────────────────────────────────────────

def fedavg(states: list, weights: "list | None" = None) -> "dict | None":
    """Average of a list of client/server state dicts, weighted by ``weights`` (e.g. the
    clients' sample counts, as FedAvg's n_k / n), equal weights when ``None`` or all equal.

    Each state dict maps slice_index → {"model": state_dict, "optimizer": ...}.
    Optimizer state is dropped (not averaged — re-warms on the next batch).
    """
    if weights is not None:
        if len(weights) != len(states):
            raise ValueError(f"fedavg: {len(weights)} weights for {len(states)} states")
        weights = [w for s, w in zip(states, weights) if s is not None]
    states = [s for s in states if s is not None]
    if not states:
        return None
    n = len(states)
    if n == 1:
        return copy.deepcopy(states[0])
    if weights is not None and len(set(weights)) > 1:
        total = float(sum(weights))
        w = torch.tensor([x / total for x in weights], dtype=torch.float64)
        result = {}
        for idx in states[0]:
            model_sd = states[0][idx]["model"]
            avg = {}
            for k in model_sd:
                stacked = torch.stack([s[idx]["model"][k].double() for s in states])
                avg[k] = (stacked * w.reshape((-1,) + (1,) * (stacked.dim() - 1))).sum(0).to(model_sd[k].dtype)
            result[idx] = {"model": avg, "optimizer": None}
        return result
    result = {}
    for idx in states[0]:
        model_sd = states[0][idx]["model"]
        avg = {}
        for k in model_sd:
            stacked = torch.stack([s[idx]["model"][k].float() for s in states])
            avg[k] = stacked.mean(0).to(model_sd[k].dtype)
        result[idx] = {"model": avg, "optimizer": None}
    return result


# ── DefaultAlgorithm ──────────────────────────────────────────────────────────

class DefaultAlgorithm(Algorithm):
    """Vanilla SplitNN. Reproduces today's behaviour bit-for-bit."""


register("default", DefaultAlgorithm)


# ── SplitFedAlgorithm (V2) ─────────────────────────────────────────────────

class SplitFedAlgorithm(Algorithm):
    """SplitFed V2 (Thapa et al., AAAI 2022): single shared server model;
    FedAvg on client-side models only after each round.

    Works with a ``ts.ClientSession(mode="parallel")``; this algorithm
    provides the aggregation in place of a manual ``on_round_end`` callback.

    ``weighting``: "samples" (default) averages the clients by their sample counts
    n_k / n, as in the paper's Algorithm 1; "equal" ignores the sizes.

    Reference: arXiv:2004.12088 / https://ojs.aaai.org/index.php/AAAI/article/view/20825
    """

    def __init__(self, weighting: str = "samples"):
        if weighting not in ("samples", "equal"):
            raise ValueError("weighting must be 'samples' or 'equal'")
        self.weighting = weighting
        self._weights = None

    def on_round_end(self, client_states, metrics, round):
        self._weights = metrics.get("client_samples") if self.weighting == "samples" else None
        avg = fedavg(client_states, self._weights)
        if avg is None:
            return None
        return [avg] * len(client_states)

    def aggregate(self, client_states, server_states, round):
        return fedavg(client_states, self._weights), None


register("splitfed", SplitFedAlgorithm)


# ── SplitFedV1Algorithm ────────────────────────────────────────────────────────

class SplitFedV1Algorithm(Algorithm):
    """SplitFed V1 (Thapa et al., AAAI 2022): per-client server copies;
    FedAvg on both client-side and server-side models after each round.

    Requires the executor to allocate separate server-slice states per client
    (``needs_per_client_server_state()`` returns True).

    ``weighting``: "samples" (default) averages the clients by their sample counts
    n_k / n, as in the paper's Algorithm 1; "equal" ignores the sizes.

    Reference: arXiv:2004.12088 / https://ojs.aaai.org/index.php/AAAI/article/view/20825
    """

    def needs_per_client_server_state(self) -> bool:
        return True

    def __init__(self, weighting: str = "samples"):
        if weighting not in ("samples", "equal"):
            raise ValueError("weighting must be 'samples' or 'equal'")
        self.weighting = weighting
        self._weights = None

    def on_round_end(self, client_states, metrics, round):
        self._weights = metrics.get("client_samples") if self.weighting == "samples" else None
        avg = fedavg(client_states, self._weights)
        if avg is None:
            return None
        return [avg] * len(client_states)

    def aggregate(self, client_states, server_states, round):
        w = self._weights if self._weights is not None and len(self._weights) == len(server_states) else None
        return fedavg(client_states, self._weights), fedavg(server_states, w)


register("splitfed_v1", SplitFedV1Algorithm)


# ── LocalLossAlgorithm ────────────────────────────────────────────────────────

class LocalLossAlgorithm(Algorithm):
    """Decoupled SL with a client-side auxiliary loss.

    The client trains layers 0..cut_slice with an auxiliary classifier at
    the cut; no gradient flows from the server.  Only forward activations
    cross the cut (or none, since the server slices are never run).

    Reference: "Decoupled Split Learning via Auxiliary Loss", 2026,
    arXiv:2601.19261.

    Args:
        aux_head: nn.Module that maps the cut activation to class logits.
            Moved to the training device at configure() time.
        cut_slice_idx: index of the last client-side slice. None = last
            slice in placement.client_slices (or 0 if no SL setting).
        aux_lr: learning rate for the auxiliary head's Adam optimizer.
    """

    def __init__(
        self,
        aux_head: nn.Module,
        cut_slice_idx: "int | None" = None,
        aux_lr: float = 1e-3,
    ):
        self.aux_head = aux_head
        self._cut_idx = cut_slice_idx
        self._aux_lr = aux_lr
        self._aux_optimizer: "optim.Optimizer | None" = None

    def configure(self, n_slices, criterion, placement):
        super().configure(n_slices, criterion, placement)
        if self._criterion is None:
            import torch.nn as _nn
            self._criterion = _nn.CrossEntropyLoss()
        self._aux_optimizer = optim.Adam(self.aux_head.parameters(), lr=self._aux_lr)

    def backward_policy(self, round: int) -> BackwardPolicy:
        return BackwardPolicy.LOCAL_LOSS

    def client_cut(self) -> "int | None":
        if self._cut_idx is not None:
            return self._cut_idx
        client = sorted(getattr(self._placement, "client_slices", []))
        return client[-1] if client else 0

    def client_loss(self, cut_output, labels, ctx):
        self.aux_head = self.aux_head.to(cut_output.device)
        logits = self.aux_head(cut_output)
        return self._criterion(logits, labels)

    def step_aux(self, ctx):
        if self._aux_optimizer is not None:
            self._aux_optimizer.step()
            self._aux_optimizer.zero_grad()


# LocalLossAlgorithm is NOT auto-registered: it requires an aux_head argument
# and cannot be instantiated by name alone. Use it directly:
#   algorithm=ts.LocalLossAlgorithm(aux_head=nn.Linear(D_H, C))
