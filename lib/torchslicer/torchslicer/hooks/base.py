"""
ActivationHook — intercept smashed activations at partition boundaries.

A hook is called on the output tensor of each non-final partition (the
"smashed data" — the activation crossing the cut layer) and can:

1. **Transform it** — add noise, compress, encrypt, etc.
   Override ``on_forward_smash`` and return the modified tensor.

2. **Contribute an auxiliary loss** — e.g. a distance-correlation penalty
   to discourage the smashed data from being correlated with raw inputs.
   Accumulate state in ``on_forward_smash`` and return a loss tensor
   from ``pop_aux_loss``.  The schedule adds this to the main loss before
   the backward pass.  ``pop_aux_loss`` must reset internal state so
   successive batches are independent.

Hooks are attached to a ``BasePipelineSchedule`` and apply to all
intermediate partition outputs in order::

    import torchslicer as ts

    schedule = ts.StandardSchedule(hooks=[
        ts.DPNoiseHook(sigma=0.01),
        ts.NoPeekHook(lambda_=0.1),
    ])
    sliced = ts.slice(model, n=3, executor=ts.LocalExecutor(schedule=schedule))
    sliced.train(loader, optimizer="adam", criterion="cross_entropy")

Hooks are called on every non-final partition boundary (i.e. after partition 0,
after partition 1, …, but NOT after the last partition whose output is the
model prediction fed directly to the criterion).
"""

from abc import ABC

import torch


class ActivationHook(ABC):
    """Base class for cut-point hooks in split learning.

    A hook is called at every non-final partition boundary — the point where
    smashed activations cross from one partition to the next.  Override any
    combination of the three methods below; all have safe pass-through defaults
    so you only implement what you need.

    Forward interception
    --------------------
    ``on_forward_smash`` receives the outgoing activation tensor and can
    transform it (add noise, compress, encrypt, …) or just observe it::

        class MyForwardHook(ActivationHook):
            def on_forward_smash(self, smashed, raw_input, partition_idx):
                return smashed + torch.randn_like(smashed) * 0.1

    Auxiliary loss
    --------------
    ``pop_aux_loss`` lets a hook contribute a scalar loss term that is added to
    the main criterion loss before the backward pass.  Accumulate state in
    ``on_forward_smash`` and return (then reset) the loss tensor here::

        class PenaltyHook(ActivationHook):
            def on_forward_smash(self, smashed, raw_input, partition_idx):
                self._penalty = some_metric(smashed, raw_input)
                return smashed

            def pop_aux_loss(self):
                loss, self._penalty = self._penalty, None
                return loss

    Backward / gradient interception
    ---------------------------------
    ``on_backward_smash`` receives the gradient tensor flowing *back* through
    the same cut point during the backward pass.  Use this for gradient
    compression, sparsification, noise injection on the gradient signal, or
    studying gradient leakage::

        class GradientClipHook(ActivationHook):
            def on_backward_smash(self, grad, partition_idx):
                return grad.clamp(-1.0, 1.0)

    The ``partition_idx`` is the same index in both directions: index *i* is
    the cut between partition *i* and partition *i+1*.
    """

    def on_forward_smash(
        self,
        smashed: "torch.Tensor",
        raw_input: "torch.Tensor",
        partition_idx: int,
    ) -> "torch.Tensor":
        """Called after partition *partition_idx* forward pass.

        Args:
            smashed:       Output of partition *partition_idx* (the smashed activation).
            raw_input:     Original batch input to partition 0.
            partition_idx: Cut-point index.

        Returns:
            The (possibly modified) smashed tensor passed to the next partition.
        """
        return smashed

    def on_backward_smash(
        self,
        grad: "torch.Tensor",
        partition_idx: int,
    ) -> "torch.Tensor":
        """Called on the gradient flowing backward through cut point *partition_idx*.

        This is the gradient of the loss w.r.t. the smashed activation — the
        signal that partition *partition_idx* uses to update its parameters.
        Override to compress, sparsify, clip, add noise, or log this gradient.

        Args:
            grad:          Gradient tensor at cut point *partition_idx*.
            partition_idx: Cut-point index (same as the corresponding forward hook).

        Returns:
            The (possibly modified) gradient passed to partition *partition_idx*'s backward.
        """
        return grad

    def pop_aux_loss(self) -> "torch.Tensor | None":
        """Return an accumulated auxiliary loss tensor, then reset state.

        Called once per batch, after all partition forwards but before the
        backward pass.  Return ``None`` (default) if this hook has no auxiliary loss.
        """
        return None
