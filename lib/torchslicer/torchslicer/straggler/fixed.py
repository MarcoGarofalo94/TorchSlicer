"""
FixedDelayPolicy — inject deterministic delays per partition.

Useful for modelling heterogeneous hardware where partition i is known to be
slower (e.g. it sits on a weaker GPU or a high-latency network link).

Example::

    from torchslicer.straggler import FixedDelayPolicy
    from torchslicer.pipeline import StandardSchedule
    from torchslicer.executors.local import LocalExecutor

    # Partition 1 is always 80 ms slower on forward, 40 ms on backward
    policy = FixedDelayPolicy(
        forward_delays  = {1: 0.08},
        backward_delays = {1: 0.04},
    )
    schedule = StandardSchedule(straggler_policy=policy)
    executor = LocalExecutor(schedule=schedule)
"""

from .base import StragglerPolicy


class FixedDelayPolicy(StragglerPolicy):
    """Inject deterministic, per-partition delays.

    Args:
        forward_delays:  Dict mapping partition index → delay in seconds
                         added after that partition's forward pass.
        backward_delays: Dict mapping partition index → delay in seconds
                         added after that partition's backward pass.

    Pass only the partitions you want to slow down; all others are unaffected.
    You can also pass a single ``delays`` dict that applies to both phases::

        # Same delay for forward and backward
        policy = FixedDelayPolicy(delays={0: 0.05, 2: 0.1})

        # Different delays per phase
        policy = FixedDelayPolicy(forward_delays={1: 0.1}, backward_delays={1: 0.02})
    """

    def __init__(
        self,
        delays: dict = None,
        forward_delays: dict = None,
        backward_delays: dict = None,
    ):
        self._fwd = dict(forward_delays or delays or {})
        self._bwd = dict(backward_delays or delays or {})

    def delay(self, partition_idx: int, phase: str) -> float:
        if phase == "forward":
            return float(self._fwd.get(partition_idx, 0.0))
        return float(self._bwd.get(partition_idx, 0.0))
