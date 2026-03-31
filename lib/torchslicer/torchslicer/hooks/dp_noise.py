"""
DPNoiseHook — Gaussian noise injection on smashed activations.

Adds calibrated Gaussian noise to the cut-layer activation before it is
passed to the next partition.  This provides a simple local differential-
privacy guarantee: an eavesdropper observing the noisy smashed data cannot
easily reconstruct the raw input features.

The noise scale should be chosen relative to the L2 sensitivity of the
activation and the desired (ε, δ)-DP guarantee.  A common starting point
for experimentation is ``sigma=0.01`` to ``0.1`` (relative to the typical
activation magnitude).

Reference
---------
Geyer et al., "Differentially Private Federated Learning: A Client Level
Perspective", 2017.  (The same noise-addition mechanism applies to smashed
data in split learning.)
"""

import torch

from .base import ActivationHook


class DPNoiseHook(ActivationHook):
    """Add independent Gaussian noise to each smashed activation tensor.

    Noise is added only during training (``model.training == True``).
    At evaluation time the hook is a no-op so inference accuracy is
    unaffected.

    Args:
        sigma:      Standard deviation of the Gaussian noise added to the
                    smashed data (default 0.1).  Scale relative to the
                    expected activation magnitude.
        partition_indices:
                    If provided, only add noise after the listed partition
                    indices.  ``None`` (default) applies noise after every
                    non-final partition boundary.

    Example::

        schedule = ts.StandardSchedule(hooks=[ts.DPNoiseHook(sigma=0.05)])
        sliced   = ts.slice(model, n=3,
                            executor=ts.LocalExecutor(schedule=schedule))
        sliced.train(loader, optimizer="adam", criterion="cross_entropy")
    """

    def __init__(self, sigma: float = 0.1, partition_indices: list[int] | None = None):
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        self.sigma             = sigma
        self.partition_indices = (set(partition_indices)
                                  if partition_indices is not None else None)

    def on_forward_smash(self, smashed: torch.Tensor, raw_input: torch.Tensor,
                         partition_idx: int) -> torch.Tensor:
        if not smashed.requires_grad and not torch.is_grad_enabled():
            return smashed  # eval / no-grad context
        if self.partition_indices is not None and partition_idx not in self.partition_indices:
            return smashed
        if self.sigma == 0.0:
            return smashed
        noise = torch.randn_like(smashed) * self.sigma
        return smashed + noise
