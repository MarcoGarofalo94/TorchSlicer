"""
PipelineTopology — sequential split learning (the TorchSlicer default).

Data flows linearly through all partitions; the last partition computes the
loss and holds the labels.

    Client ──► Server0 ──► Server1 ──► … ──► ServerN-1 (loss)
         ◄──────────────────────────────── gradient

All partitions can conceptually belong to different workers.  There is no
role distinction between "client" and "server" at this level — every
participant is a pipeline stage.
"""

from .base import BaseSplitTopology


class PipelineTopology(BaseSplitTopology):
    """Sequential pipeline: partitions 0, 1, …, N-1, loss at partition N-1.

    This formalises the topology that TorchSlicer has always used by default.
    Attaching it to a ``SlicedModel`` makes the design intent explicit in logs
    and enables future distributed routing to use the same interface as other
    topologies::

        sliced = ts.slice(model, n=4)
        sliced.topology = ts.PipelineTopology()
    """

    def assign(self, partitions: list) -> dict[str, list]:
        return {"stage": list(partitions)}

    def loss_owner(self, partitions: list) -> str:
        return "stage"

    def describe(self) -> str:
        return "stage_0 → stage_1 → … → stage_N (loss)"
