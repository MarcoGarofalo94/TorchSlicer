"""
GradientSparsifyHook — TopK gradient sparsification at cut points.

Based on Deep Gradient Compression (Lin et al., ICLR 2018), which shows that
99% of gradient entries can be dropped with negligible convergence loss when
momentum correction is applied.  This implementation provides the core
sparsification primitive; momentum correction and error feedback are left to
the researcher to layer on top via a custom subclass.

Example::

    from torchslicer.hooks import GradientSparsifyHook
    from torchslicer.pipeline import StandardSchedule
    from torchslicer.executors.local import LocalExecutor

    # Keep only the top 1% of gradient values at every cut point
    schedule = StandardSchedule(hooks=[GradientSparsifyHook(keep_ratio=0.01)])
    executor = LocalExecutor(schedule=schedule)
    history = ts.run(model, loader, n=3, executor=executor, optimizer="adamw")

    # Apply sparsification only at cut point 0 (between partition 0 and 1)
    schedule = StandardSchedule(hooks=[GradientSparsifyHook(keep_ratio=0.1,
                                                            partition_indices=[0])])
"""

import torch

from .base import ActivationHook


class GradientSparsifyHook(ActivationHook):
    """TopK gradient sparsification at split-layer cut points.

    Keeps the ``keep_ratio`` fraction of gradient entries with the largest
    absolute values and zeros out the rest.  This simulates gradient
    compression in split learning — the upstream partition only receives a
    sparse gradient signal, reducing effective communication bandwidth.

    This is a forward no-op: ``on_forward_smash`` returns the activation
    unchanged.  All the action is in ``on_backward_smash``.

    Args:
        keep_ratio:        Fraction of gradient entries to keep (default 0.01
                           → top 1%).  Clamped so at least one entry survives.
        partition_indices: If given, only apply at these cut-point indices.
                           ``None`` (default) applies at every cut point.

    Reference:
        Lin et al., "Deep Gradient Compression", ICLR 2018.
        https://arxiv.org/abs/1712.01887
    """

    def __init__(self, keep_ratio: float = 0.01, partition_indices=None):
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        self.keep_ratio = keep_ratio
        self.partition_indices = set(partition_indices) if partition_indices is not None else None

    def on_backward_smash(self, grad: torch.Tensor, partition_idx: int) -> torch.Tensor:
        if self.partition_indices is not None and partition_idx not in self.partition_indices:
            return grad
        flat = grad.flatten()
        k = max(1, int(flat.numel() * self.keep_ratio))
        _, top_indices = flat.abs().topk(k, sorted=False)
        mask = torch.zeros_like(flat)
        mask[top_indices] = 1.0
        return (flat * mask).reshape(grad.shape)
