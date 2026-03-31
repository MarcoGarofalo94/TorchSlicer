"""
ParameterBalancedSplitter — partition by trainable parameter count.

Assigns layers to partitions greedily so that each partition holds roughly
the same number of trainable parameters.  Layers with zero trainable
parameters (e.g. BatchNorm in eval, frozen embeddings) count as 0 and are
folded into whichever partition is currently lightest.

When all layers have zero parameters (e.g. a pure-activation network) the
splitter falls back to uniform distribution.
"""

from .base import BaseSplitter, Partition
from .uniform import UniformSplitter


def _param_count(layer) -> int:
    try:
        return sum(p.numel() for p in layer.parameters() if p.requires_grad)
    except Exception:
        return 0


class ParameterBalancedSplitter(BaseSplitter):
    """Split a model so each partition has a roughly equal parameter count.

    Uses a greedy left-to-right bin-filling pass: layers are assigned
    sequentially to the current partition until it would exceed the per-
    partition target, then a new partition is opened.

    This keeps layers contiguous (a requirement of the current executor)
    while minimising parameter imbalance.  It is most useful for models
    where layer sizes vary significantly — e.g. a transformer whose
    embedding and head layers are much larger than the blocks.

    Example::

        sliced = ts.slice(model, n=4, strategy="param_balanced")
    """

    def split(self, model_graph, n_partitions: int) -> list:
        layers = model_graph.get_layers()
        n_layers = len(layers)

        if n_partitions >= n_layers:
            return UniformSplitter().split(model_graph, n_partitions)

        counts = [_param_count(l) for l in layers]
        total  = sum(counts)

        # Fall back to uniform when all layers are parameter-free
        if total == 0:
            return UniformSplitter().split(model_graph, n_partitions)

        target    = total / n_partitions
        partitions = []
        current_indices:  list[int] = []
        current_params:   int       = 0
        partition_index:  int       = 0

        for i, cnt in enumerate(counts):
            current_indices.append(i)
            current_params += cnt

            remaining_layers     = n_layers - i - 1
            remaining_partitions = n_partitions - partition_index - 1

            # Open a new partition if we hit the per-partition target AND enough
            # layers remain to fill all remaining partitions (≥1 each).
            if (current_params >= target
                    and remaining_layers >= remaining_partitions
                    and partition_index < n_partitions - 1):
                preds = UniformSplitter._intra_predecessors(model_graph, current_indices)
                partitions.append(Partition(
                    index=partition_index,
                    layer_indices=list(current_indices),
                    predecessors=preds,
                ))
                partition_index += 1
                current_indices = []
                current_params  = 0

        # Flush remaining layers into the last partition
        if current_indices:
            preds = UniformSplitter._intra_predecessors(model_graph, current_indices)
            partitions.append(Partition(
                index=partition_index,
                layer_indices=list(current_indices),
                predecessors=preds,
            ))

        return partitions
