"""
BaseSplitTopology — describes how model partitions are assigned to participants.

A topology answers three questions:
  1. Which partitions belong to which role (client / server / etc.)?
  2. Which participant computes the loss?
  3. What is the conceptual data-flow between participants?

In **local** execution all participants share one process so the topology only
provides metadata — the execution order is always sequential regardless of
topology.  In future **distributed** topologies, the role assignment will
drive gRPC routing.

Built-in topologies
-------------------
- ``PipelineTopology``  — sequential pipeline, labels on last partition.
  This is the implicit default for all existing ``ts.slice()`` calls.
- ``UShapedTopology``   — client holds the first and last partition(s),
  server holds the middle; server never sees labels.

Attaching a topology to a SlicedModel (metadata only, does not change
local execution)::

    sliced = ts.slice(model, n=3)
    sliced.topology = ts.UShapedTopology()
    roles = sliced.topology.assign(sliced.partitions)
    # {"client": [p0, p2], "server": [p1]}
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class BaseSplitTopology(ABC):
    """Abstract base for split-learning topologies.

    Subclasses define how *n* sequential partitions map to participant roles.
    """

    @abstractmethod
    def assign(self, partitions: list) -> dict[str, list]:
        """Return ``{role: [Partition, ...]}`` for the given partition list.

        The role names (e.g. ``"client"``, ``"server"``) are topology-specific
        strings intended for logging and future routing logic.
        """
        ...

    @abstractmethod
    def loss_owner(self, partitions: list) -> str:
        """Return the role name of the participant that computes the loss."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """One-line human-readable description of the data flow."""
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.describe()})"
