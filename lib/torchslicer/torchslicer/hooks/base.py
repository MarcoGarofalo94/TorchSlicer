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

from abc import ABC, abstractmethod

import torch


class ActivationHook(ABC):
    """Abstract base class for smashed-data activation hooks.

    Subclasses must implement ``on_forward_smash``.  ``pop_aux_loss`` is
    optional — override it when the hook needs to contribute a loss term.
    """

    @abstractmethod
    def on_forward_smash(
        self,
        smashed: "torch.Tensor",
        raw_input: "torch.Tensor",
        partition_idx: int,
    ) -> "torch.Tensor":
        """Called after partition *partition_idx* forward pass.

        Args:
            smashed:       Output tensor of partition *partition_idx*
                           (the smashed / cut-layer activation).
            raw_input:     The original batch input (partition 0's input).
                           Useful for privacy hooks that measure correlation
                           between raw features and intermediate activations.
            partition_idx: Index of the partition whose output this is.

        Returns:
            The (possibly modified) smashed tensor passed to the next partition.
        """
        ...

    def pop_aux_loss(self) -> "torch.Tensor | None":
        """Return an accumulated auxiliary loss tensor, then reset state.

        Called once per batch, after all partition forwards but before the
        backward pass.  The returned tensor is added to the main criterion
        loss so gradients flow through it automatically.

        Return ``None`` (default) if this hook has no auxiliary loss.
        """
        return None
