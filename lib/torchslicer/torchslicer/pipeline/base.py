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

from abc import ABC, abstractmethod


class BasePipelineSchedule(ABC):
    """Abstract base for pipeline schedules.

    A schedule receives a prepared batch and the list of ``SplitLayer``
    objects (already set up with optimizers) and is responsible for running
    the complete forward + backward + optimizer step.

    Args (passed to ``step()``):
        split_layers: Ordered list of ``SplitLayer`` instances (partition 0
                      first, last partition last).
        inputs:       Main input tensor, already on the correct device.
        labels:       Target tensor, already on the correct device.
        aux:          Dict of auxiliary inputs forwarded to partition 0.
        criterion:    Loss function (``nn.Module``).
        profilers:    Per-partition ``WorkerProfiler`` list (same length as
                      ``split_layers``).  Use ``profiler.phase(name)`` as a
                      context manager to record timing.
        batch_id:     Global batch index for tracing / logging.

    Returns:
        Scalar loss value (Python ``float``) for the batch.
    """

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
    ) -> float: ...
