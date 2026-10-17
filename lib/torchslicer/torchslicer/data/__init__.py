"""Data for SL experiments: federated partitioners (M11)."""
from .partition import (
    ClientSplit,
    available_partitioners,
    labels_of,
    partition,
    register_partitioner,
)

__all__ = ["ClientSplit", "available_partitioners", "labels_of", "partition", "register_partitioner"]
