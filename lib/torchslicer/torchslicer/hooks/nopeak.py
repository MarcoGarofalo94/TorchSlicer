"""
NoPeekHook — distance-correlation penalty on smashed activations.

Minimises the statistical dependence between raw inputs and the smashed
(cut-layer) activations by adding a distance-correlation (dCor) penalty to
the training loss.  When the penalty is large enough, the smashed data
becomes difficult to use to reconstruct the raw inputs.

Reference
---------
Vepakomma et al., "NoPeek: Information leakage reduction to share
activations in distributed deep learning", ICDMW 2020.
https://arxiv.org/abs/2008.03156

Distance correlation
--------------------
For two random vectors A and B with n samples:

    1. Compute pairwise Euclidean distance matrices D_A, D_B  (n×n).
    2. Double-centre each matrix: Â = D_A - row_mean - col_mean + grand_mean.
    3. dCov²(A,B)  = mean(Â ⊙ B̂)
    4. dCor(A,B)   = dCov(A,B) / sqrt(dVar(A) · dVar(B))

The penalty is ``lambda_ * dCor(X, Z)`` where X is the raw input batch and Z
is the smashed activation.  dCor is in [0, 1]; the optimizer minimises it
while also minimising task loss.

Usage::

    schedule = ts.StandardSchedule(hooks=[ts.NoPeekHook(lambda_=0.1)])
    sliced   = ts.slice(model, n=3,
                        executor=ts.LocalExecutor(schedule=schedule))
    sliced.train(loader, optimizer="adamw", criterion="cross_entropy")

Notes
-----
- Complexity is O(B²) in batch size B.  Use batch sizes ≤ 256 for tractable
  training.
- ``lambda_`` controls the privacy–utility tradeoff.  Start at 0.05–0.1 and
  tune.  Too high → model underfits; too low → privacy gain is negligible.
- Only applied to the first non-final partition boundary by default
  (``partition_indices=[0]``).  This is the most informative cut point
  because it sits closest to raw inputs.
"""

import torch

from .base import ActivationHook


def _distance_matrix(x: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distance matrix for a batch [B, *]."""
    x_flat = x.flatten(start_dim=1)          # [B, d]
    diff    = x_flat.unsqueeze(0) - x_flat.unsqueeze(1)  # [B, B, d]
    return torch.norm(diff, dim=-1)           # [B, B]


def _double_center(d: torch.Tensor) -> torch.Tensor:
    """Double-centre a distance matrix."""
    row_mean   = d.mean(dim=1, keepdim=True)
    col_mean   = d.mean(dim=0, keepdim=True)
    grand_mean = d.mean()
    return d - row_mean - col_mean + grand_mean


def distance_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the sample distance correlation between tensors a and b.

    Args:
        a: Tensor of shape [B, *] (batch dimension first).
        b: Tensor of shape [B, *] (same batch size as a).

    Returns:
        Scalar tensor in [0, 1].  Returns 0 if either variable has zero
        distance variance (i.e. is constant within the batch).
    """
    Da = _double_center(_distance_matrix(a))
    Db = _double_center(_distance_matrix(b))

    dcov2_ab = (Da * Db).mean()
    dcov2_a  = (Da * Da).mean()
    dcov2_b  = (Db * Db).mean()

    # dCor²(A,B) = dCov²(A,B) / sqrt(dVar²(A) · dVar²(B))
    # dCor(A,B)  = sqrt(dCor²(A,B))
    denom = (dcov2_a * dcov2_b).clamp(min=0.0).sqrt()
    if denom.item() == 0.0:
        return torch.tensor(0.0, device=a.device, dtype=a.dtype)

    dcor2 = (dcov2_ab / denom).clamp(min=0.0)
    return dcor2.sqrt()


class NoPeekHook(ActivationHook):
    """Add a distance-correlation penalty to discourage smashed-data leakage.

    Args:
        lambda_:           Weight of the dCor penalty in the total loss
                           (default 0.1).  Higher → more privacy, potentially
                           lower task accuracy.
        partition_indices: Apply penalty only after these partition indices.
                           Defaults to ``[0]`` (only the first cut point,
                           closest to raw inputs).

    Example::

        hooks    = [ts.NoPeekHook(lambda_=0.05)]
        schedule = ts.StandardSchedule(hooks=hooks)
        sliced   = ts.slice(model, n=3,
                            executor=ts.LocalExecutor(schedule=schedule))
        sliced.train(loader, optimizer="adamw", criterion="cross_entropy")
    """

    def __init__(self, lambda_: float = 0.1, partition_indices: list[int] | None = None):
        if lambda_ < 0:
            raise ValueError(f"lambda_ must be non-negative, got {lambda_}")
        self.lambda_           = lambda_
        self.partition_indices = (set(partition_indices)
                                  if partition_indices is not None else {0})
        self._pending: torch.Tensor | None = None

    def on_forward_smash(self, smashed: torch.Tensor, raw_input: torch.Tensor,
                         partition_idx: int) -> torch.Tensor:
        if partition_idx not in self.partition_indices:
            return smashed
        if self.lambda_ == 0.0:
            return smashed

        dcor = distance_correlation(raw_input, smashed)
        penalty = self.lambda_ * dcor

        # Accumulate (multiple boundaries may contribute)
        self._pending = penalty if self._pending is None else self._pending + penalty
        return smashed  # NoPeek does NOT modify the smashed tensor itself

    def pop_aux_loss(self) -> torch.Tensor | None:
        loss = self._pending
        self._pending = None
        return loss
