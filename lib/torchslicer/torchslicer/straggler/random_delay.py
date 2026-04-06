"""
RandomDelayPolicy — probabilistic straggler injection.

Models bursty or intermittent stragglers: each partition independently has a
``prob`` chance of being delayed by up to ``max_delay_s`` seconds on any given
forward or backward pass.

Example::

    from torchslicer.straggler import RandomDelayPolicy

    # Any partition, on any pass, has a 20% chance of a 0–200 ms delay
    policy = RandomDelayPolicy(prob=0.2, max_delay_s=0.2)

    # Only partition 2 can straggle, forward only
    policy = RandomDelayPolicy(prob=0.3, max_delay_s=0.5,
                               partition_indices=[2], phases=["forward"])
"""

import random

from .base import StragglerPolicy


class RandomDelayPolicy(StragglerPolicy):
    """Randomly inject delays to simulate intermittent stragglers.

    Each call to ``delay()`` independently draws from Uniform(0, max_delay_s)
    with probability ``prob``, and returns 0.0 otherwise.

    Args:
        prob:              Probability of a delay occurring (default 0.1).
        max_delay_s:       Maximum delay in seconds (default 0.1).  The actual
                           delay is sampled uniformly from [0, max_delay_s].
        partition_indices: Restrict straggling to these partition indices.
                           ``None`` (default) applies to all partitions.
        phases:            Restrict to ``"forward"``, ``"backward"``, or both
                           (default ``None`` → both phases).
        seed:              Optional random seed for reproducibility.
    """

    def __init__(
        self,
        prob: float = 0.1,
        max_delay_s: float = 0.1,
        partition_indices=None,
        phases=None,
        seed: int = None,
    ):
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"prob must be in [0, 1], got {prob}")
        self.prob = prob
        self.max_delay_s = max_delay_s
        self.partition_indices = set(partition_indices) if partition_indices is not None else None
        self.phases = set(phases) if phases is not None else None
        self._rng = random.Random(seed)

    def delay(self, partition_idx: int, phase: str) -> float:
        if self.partition_indices is not None and partition_idx not in self.partition_indices:
            return 0.0
        if self.phases is not None and phase not in self.phases:
            return 0.0
        if self._rng.random() < self.prob:
            return self._rng.uniform(0.0, self.max_delay_s)
        return 0.0
