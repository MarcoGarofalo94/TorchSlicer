from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Partition:
    index: int
    layer_indices: list
    # For each layer: list of partition-local predecessor indices.
    # Empty list → layer takes this partition's external input.
    predecessors: list = field(default_factory=list)


class BaseSplitter(ABC):
    """Turns a ``ModelGraph`` into ``Partition``s.

    Subclasses implement ``split``. The easiest way is to pick cut positions from
    ``model_graph.valid_cuts()`` and return ``self.partitions_from_cuts(model_graph, cuts)``.
    See docs/tutorials/custom_strategy.md.
    """

    @abstractmethod
    def split(self, model_graph, n_partitions: int) -> list:
        ...

    def expand_for(self, model) -> list:
        """Submodules that must be traced through for this splitter (see ``ModelGraph.from_module``)."""
        return []

    # ── helpers for subclasses ────────────────────────────────────────────────

    @classmethod
    def partitions_from_cuts(cls, model_graph, cuts) -> list:
        """Partitions for cuts after the given node indices (sorted, each in ``valid_cuts()``)."""
        bounds = [-1, *sorted(cuts), len(model_graph) - 1]
        partitions = []
        for i in range(len(bounds) - 1):
            layer_indices = list(range(bounds[i] + 1, bounds[i + 1] + 1))
            partitions.append(Partition(
                index=i,
                layer_indices=layer_indices,
                predecessors=cls._intra_predecessors(model_graph, layer_indices),
            ))
        return partitions

    @staticmethod
    def _intra_predecessors(model_graph, layer_indices: list) -> list:
        """
        For each layer in the partition, compute which other layers *within the
        same partition* feed into it.

        Returns list[list[int]] where result[i] is a list of partition-local
        indices of predecessors.  Empty list = takes the partition's external
        input (from data loader or the previous partition's output). A
        multi-input layer lists ``-1`` where it takes that external input
        (e.g. a residual add right after a cut: ``[-1, 3]``).
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
            if len(node.predecessors) > 1:
                local_preds = [global_to_local.get(p, -1) for p in node.predecessors]
            else:
                local_preds = [global_to_local[p] for p in node.predecessors if p in index_set]
            result.append(local_preds)

        return result

    # ── validation ────────────────────────────────────────────────────────────

    def validate(self, model_graph, partitions: list) -> None:
        """Reject partitions the executors would run incorrectly.

        Each partition receives exactly one tensor, the output of the previous
        partition's last layer, so: partitions are non-empty, contiguous and in
        order; only the model's first partition reads the model input; and every
        edge that crosses a boundary goes from the last layer of one partition to
        the next partition.
        """
        # All layers covered
        all_indices = set()
        for p in partitions:
            for idx in p.layer_indices:
                all_indices.add(idx)
        expected = set(range(len(model_graph)))
        if all_indices != expected:
            raise ValueError(f"Partitions don't cover all layers: missing {expected - all_indices}")

        if not model_graph.nodes:
            return

        flat = [idx for p in partitions for idx in p.layer_indices]
        if any(not p.layer_indices for p in partitions) or flat != list(range(len(model_graph))):
            raise ValueError(
                "Partitions must be non-empty, contiguous runs of layer indices in order "
                f"(0..{len(model_graph) - 1}); got {[list(p.layer_indices) for p in partitions]}"
            )

        node_to_partition = {}
        for pos, p in enumerate(partitions):
            for idx in p.layer_indices:
                node_to_partition[idx] = pos
        last_of = [p.layer_indices[-1] for p in partitions]
        hint = self._cut_hint(model_graph)

        for node in model_graph.nodes:
            pos = node_to_partition[node.index]
            if node.takes_input and pos != 0:
                raise ValueError(
                    f"Node {node.index} ({node.name or type(node.module).__name__}) reads the model "
                    f"input but is in partition {pos}; only partition 0 receives it.{hint}"
                )
            for q in node.predecessors:
                q_pos = node_to_partition[q]
                if q_pos == pos or (q_pos == pos - 1 and q == last_of[q_pos]):
                    continue
                if len(node.predecessors) > 1:
                    # e.g. a skip connection: one tensor crosses each boundary, so the
                    # other inputs of the multi-input node would be missing.
                    raise ValueError(
                        f"Node {node.index} ({type(node.module).__name__}) "
                        f"has multi-input predecessors {node.predecessors} that span partition "
                        f"boundaries. Choose a different split count so that all predecessors of "
                        f"multi-input nodes belong to the same partition.\n"
                        f"Tip: use n smaller than the number of layers before this node, or wrap "
                        f"the skip-connection block in a single nn.Module.{hint}"
                    )
                else:
                    raise ValueError(
                        f"Node {node.index} ({node.name or type(node.module).__name__}) in partition "
                        f"{pos} reads node {q} ({model_graph.nodes[q].name}) from partition {q_pos}, "
                        f"but only the output of the previous partition's last node "
                        f"({last_of[pos - 1] if pos else 'model input'}) crosses a boundary.{hint}"
                    )

    @staticmethod
    def _cut_hint(model_graph) -> str:
        names = [model_graph.nodes[k].name or str(k) for k in model_graph.valid_cuts()]
        if len(names) > 20:
            names = names[:10] + ["..."] + names[-10:]
        return f"\nValid cuts are after: {names}" if names else "\nThis graph has no valid cut points."
