import math
from .base import BaseSplitter, Partition


class UniformSplitter(BaseSplitter):
    """Equal layer *count* per partition (``ceil(len / n)`` each), ignoring layer cost."""

    def split(self, model_graph, n_partitions: int) -> list:
        n_layers = len(model_graph)
        size = math.ceil(n_layers / n_partitions)
        partitions = []
        for i in range(n_partitions):
            start = i * size
            if start >= n_layers:
                break
            end = min(start + size, n_layers)
            layer_indices = list(range(start, end))
            predecessors = self._intra_predecessors(model_graph, layer_indices)
            partitions.append(Partition(
                index=i,
                layer_indices=layer_indices,
                predecessors=predecessors,
            ))
        return partitions
