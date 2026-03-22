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
    @abstractmethod
    def split(self, model_graph, n_partitions: int) -> list:
        ...

    def validate(self, model_graph, partitions: list) -> None:
        # All layers covered
        all_indices = set()
        for p in partitions:
            for idx in p.layer_indices:
                all_indices.add(idx)
        expected = set(range(len(model_graph)))
        if all_indices != expected:
            raise ValueError(f"Partitions don't cover all layers: missing {expected - all_indices}")

        # No cross-partition multi-input nodes
        # (skip connections that span partition boundaries require protocol support
        # not yet implemented — the user must choose n so that multi-input nodes
        # remain within a single partition)
        if not model_graph.nodes:
            return
        node_to_partition = {}
        for p in partitions:
            for idx in p.layer_indices:
                node_to_partition[idx] = p.index

        for p in partitions:
            for global_idx in p.layer_indices:
                if global_idx >= len(model_graph.nodes):
                    continue
                node = model_graph.nodes[global_idx]
                if len(node.predecessors) <= 1:
                    continue
                # Multi-input node: all predecessors must be in the same or
                # earlier partition — but the current executor can only pass
                # one tensor across partition boundaries
                external_preds = [
                    pred for pred in node.predecessors
                    if node_to_partition.get(pred, -1) != p.index
                ]
                if external_preds:
                    raise ValueError(
                        f"Node {global_idx} ({type(model_graph.nodes[global_idx].module).__name__}) "
                        f"has multi-input predecessors {node.predecessors} that span partition "
                        f"boundaries. Choose a different split count so that all predecessors of "
                        f"multi-input nodes belong to the same partition.\n"
                        f"Tip: use n smaller than the number of layers before this node, or wrap "
                        f"the skip-connection block in a single nn.Module."
                    )
