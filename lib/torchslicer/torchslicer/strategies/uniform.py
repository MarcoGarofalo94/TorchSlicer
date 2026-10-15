import math
from .base import BaseSplitter, Partition


class UniformSplitter(BaseSplitter):
    """Equal layer *count* per partition (``ceil(len / n)`` each), ignoring layer cost.

    When ``ceil`` sizes would run out of layers before ``n`` partitions (4 layers, n=3:
    2 + 2), the remainder is spread instead (2 + 1 + 1), so asking for ``n`` gives ``n``.
    """

    def split(self, model_graph, n_partitions: int) -> list:
        n_layers = len(model_graph)
        if n_partitions > n_layers:
            raise ValueError(f"can't make {n_partitions} partitions from {n_layers} layers")
        size = math.ceil(n_layers / n_partitions)
        if (n_layers - 1) // size + 1 < n_partitions:
            k, r = divmod(n_layers, n_partitions)
            sizes = [k + (1 if i < r else 0) for i in range(n_partitions)]
        else:
            sizes = [size] * n_partitions
        partitions = []
        start = 0
        for i in range(n_partitions):
            if start >= n_layers:
                break
            end = min(start + sizes[i], n_layers)
            layer_indices = list(range(start, end))
            start = end
            predecessors = self._intra_predecessors(model_graph, layer_indices)
            partitions.append(Partition(
                index=i,
                layer_indices=layer_indices,
                predecessors=predecessors,
            ))
        return partitions
