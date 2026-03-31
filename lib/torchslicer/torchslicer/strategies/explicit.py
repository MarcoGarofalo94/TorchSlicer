"""
ExplicitSplitter — user-specified layer boundary indices.

Gives full control over partition boundaries without subclassing
``BaseSplitter``.  Useful when you know exactly how many layers a model has
and where the natural cut-points are (e.g. after the embedding, after every
N transformer blocks, before the head).

Example::

    # 24-layer BERT: embed | blocks 0-11 | blocks 12-23 | head
    sliced = ts.slice(model, n=4, strategy=ts.ExplicitSplitter(boundaries=[1, 13, 25]))

    # Same via the registry after registering with a name
    ts.register_strategy("my_bert_split", ts.ExplicitSplitter(boundaries=[1, 13, 25]))
    sliced = ts.slice(model, n=4, strategy="my_bert_split")
"""

from .base import BaseSplitter, Partition
from .uniform import UniformSplitter


class ExplicitSplitter(BaseSplitter):
    """Split a model at explicit layer index boundaries.

    Args:
        boundaries: Sorted list of *start* indices for each partition after
                    the first.  For a model with layers [0..N-1] and
                    ``boundaries=[k1, k2]`` the resulting partitions are:
                    ``[0..k1-1]``, ``[k1..k2-1]``, ``[k2..N-1]``.

    The ``n_partitions`` argument passed to ``split()`` must equal
    ``len(boundaries) + 1``; a ``ValueError`` is raised otherwise so that
    mismatches between the ``ts.slice(n=...)`` call and the boundary list are
    caught early.
    """

    def __init__(self, boundaries: list[int]):
        if not boundaries:
            raise ValueError("boundaries must contain at least one index")
        self.boundaries = sorted(boundaries)

    def split(self, model_graph, n_partitions: int) -> list:
        n_layers = len(model_graph)
        expected_n = len(self.boundaries) + 1
        if n_partitions != expected_n:
            raise ValueError(
                f"ExplicitSplitter has {len(self.boundaries)} boundary/ies → "
                f"{expected_n} partitions, but ts.slice() requested n={n_partitions}. "
                f"Either adjust boundaries or pass n={expected_n}."
            )

        # Build boundary list: [0, b1, b2, ..., n_layers]
        starts = [0] + list(self.boundaries) + [n_layers]
        for b in self.boundaries:
            if b <= 0 or b >= n_layers:
                raise ValueError(
                    f"Boundary {b} is out of range for a model with {n_layers} layers "
                    f"(valid range: 1 to {n_layers - 1})."
                )

        partitions = []
        for i in range(n_partitions):
            layer_indices = list(range(starts[i], starts[i + 1]))
            if not layer_indices:
                raise ValueError(
                    f"Boundary {starts[i + 1]} produces an empty partition at index {i}. "
                    f"Ensure all boundaries create at least one layer per partition."
                )
            preds = UniformSplitter._intra_predecessors(model_graph, layer_indices)
            partitions.append(Partition(
                index=i,
                layer_indices=layer_indices,
                predecessors=preds,
            ))

        return partitions
