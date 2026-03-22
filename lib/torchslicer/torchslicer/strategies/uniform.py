import math
from .base import BaseSplitter, Partition


class UniformSplitter(BaseSplitter):
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

    @staticmethod
    def _intra_predecessors(model_graph, layer_indices: list) -> list:
        """
        For each layer in the partition, compute which other layers *within the
        same partition* feed into it.

        Returns list[list[int]] where result[i] is a list of partition-local
        indices of predecessors.  Empty list = takes the partition's external
        input (from data loader or the previous partition's output).
        """
        if not model_graph.nodes:
            return [[] for _ in layer_indices]

        index_set = set(layer_indices)
        global_to_local = {g: l for l, g in enumerate(layer_indices)}
        result = []

        for global_idx in layer_indices:
            if global_idx >= len(model_graph.nodes):
                result.append([])
                continue
            node = model_graph.nodes[global_idx]
            local_preds = [
                global_to_local[p]
                for p in node.predecessors
                if p in index_set
            ]
            result.append(local_preds)

        return result
