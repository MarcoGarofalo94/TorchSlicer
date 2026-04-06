"""
StragglerPolicy — simulate slow or failing partitions in LocalExecutor.

In real distributed split learning, workers may be delayed by hardware
heterogeneity, network congestion, or competing jobs (stragglers).  This
abstraction lets researchers simulate and study straggler behaviour on a
single machine without a cluster.

A ``StragglerPolicy`` is attached to a ``BasePipelineSchedule``.  After each
partition completes its forward or backward pass the schedule calls
``delay(partition_idx, phase)`` and sleeps for the returned number of seconds.
A return value of 0.0 means no delay.

Implementing a custom policy::

    from torchslicer.straggler import StragglerPolicy

    class ExponentialDelayPolicy(StragglerPolicy):
        \"\"\"Each partition independently draws a delay from Exp(lambda).\"\"\"
        def __init__(self, lam: float = 10.0):
            self.lam = lam

        def delay(self, partition_idx: int, phase: str) -> float:
            import random
            return random.expovariate(self.lam)   # mean = 1/lam seconds

    schedule = ts.StandardSchedule(straggler_policy=ExponentialDelayPolicy(lam=5.0))
    executor = ts.LocalExecutor(schedule=schedule)
    history = ts.run(model, loader, n=4, executor=executor, optimizer="adamw")

The ``phase`` argument is either ``"forward"`` or ``"backward"``, letting you
model asymmetric stragglers (e.g. slow forward due to large compute, fast
backward because the gradient is small).
"""

from abc import ABC, abstractmethod


class StragglerPolicy(ABC):
    """Abstract base for straggler simulation policies.

    Attach to a ``BasePipelineSchedule`` via the ``straggler_policy`` argument.
    The schedule calls ``delay()`` after each partition's forward and backward
    pass and sleeps for the returned duration.

    Args to ``delay``:
        partition_idx: Index of the partition that just finished.
        phase:         ``"forward"`` or ``"backward"``.

    Returns:
        Seconds to sleep (``float``).  Return ``0.0`` for no delay.
    """

    @abstractmethod
    def delay(self, partition_idx: int, phase: str) -> float: ...
