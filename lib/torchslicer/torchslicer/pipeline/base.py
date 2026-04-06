"""
BasePipelineSchedule — abstraction for forward/backward scheduling strategies.

A pipeline schedule owns the logic for how a single batch (or set of
micro-batches) is executed across a list of ``SplitLayer`` partitions.

Implementing a custom schedule::

    import torchslicer as ts

    class MySchedule(ts.BasePipelineSchedule):
        def step(self, split_layers, inputs, labels, criterion,
                 profilers, batch_id) -> float:
            # custom forward/backward logic
            ...
            return loss_value

    sliced = ts.slice(model, n=4)
    sliced.executor.schedule = MySchedule()
    sliced.train(loader, optimizer="adam", criterion="cross_entropy")

Built-in schedules
------------------
- ``StandardSchedule``   — plain forward-then-backward, one full batch.
- ``GPipeSchedule(m)``   — GPipe all-forward / all-backward with *m* micro-batches.
"""

import time
from abc import ABC, abstractmethod


class BasePipelineSchedule(ABC):
    """Abstract base for pipeline schedules.

    Subclasses define how one batch is executed across the split layers.

    Hooks
    -----
    An optional list of ``ActivationHook`` instances can be attached at
    construction time.  Hooks are called on the output of every non-final
    partition (the "smashed data") during the forward pass, on the gradient
    at every cut point during the backward pass, and their accumulated
    ``aux_loss`` terms are added to the main criterion loss::

        schedule = ts.StandardSchedule(hooks=[
            ts.DPNoiseHook(sigma=0.01),
            ts.NoPeekHook(lambda_=0.1),
            ts.GradientSparsifyHook(keep_ratio=0.01),
        ])

    Straggler simulation
    --------------------
    A ``StragglerPolicy`` can be attached to simulate slow partitions.
    After each partition's forward or backward pass the schedule calls
    ``policy.delay(partition_idx, phase)`` and sleeps for the returned
    number of seconds.  Useful for studying straggler-mitigation algorithms
    on a single machine without a cluster::

        from torchslicer.straggler import FixedDelayPolicy

        schedule = ts.StandardSchedule(
            straggler_policy=FixedDelayPolicy(forward_delays={1: 0.05})
        )
    """

    def __init__(self, hooks: list = None, straggler_policy=None):
        self.hooks: list = list(hooks or [])
        self.straggler_policy = straggler_policy

    # ── Hook helpers ──────────────────────────────────────────────────────────

    def _apply_forward_hooks(self, smashed, raw_input, partition_idx):
        """Run all hooks' on_forward_smash. Returns modified tensor."""
        for hook in self.hooks:
            smashed = hook.on_forward_smash(smashed, raw_input, partition_idx)
        return smashed

    def _apply_backward_hooks(self, grad, partition_idx):
        """Run all hooks' on_backward_smash. Returns modified gradient."""
        if grad is None:
            return grad
        for hook in self.hooks:
            grad = hook.on_backward_smash(grad, partition_idx)
        return grad

    def _collect_aux_losses(self):
        """Collect and sum all pending aux losses from hooks. Returns Tensor or None."""
        total = None
        for hook in self.hooks:
            aux = hook.pop_aux_loss()
            if aux is not None:
                total = aux if total is None else total + aux
        return total

    # ── Straggler helper ──────────────────────────────────────────────────────

    def _straggler_delay(self, partition_idx: int, phase: str) -> None:
        """Sleep for the duration returned by the straggler policy (if any)."""
        if self.straggler_policy is None:
            return
        secs = self.straggler_policy.delay(partition_idx, phase)
        if secs > 0.0:
            time.sleep(secs)

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def step(
        self,
        split_layers: list,
        inputs,
        labels,
        aux: dict,
        criterion,
        profilers: list,
        batch_id: int,
    ) -> float:
        """Execute one full batch: forward + backward + optimizer step.

        Args:
            split_layers: Ordered list of ``SplitLayer`` instances.
            inputs:       Main input tensor.
            labels:       Target tensor.
            aux:          Dict of auxiliary inputs forwarded to partition 0.
            criterion:    Loss function (``nn.Module``).
            profilers:    Per-partition ``WorkerProfiler`` list.
            batch_id:     Global batch index for tracing / logging.

        Returns:
            Scalar loss value (Python ``float``) for the batch.
        """
        ...
