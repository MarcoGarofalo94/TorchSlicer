from .base import BaseSplitter


class NamedCutSplitter(BaseSplitter):
    """Cut right after each named node: ``NamedCutSplitter(["layer2", "layer3.1"])``.

    Names are graph node names (``ts.trace(model).summary()`` lists them) or node
    indices. A dotted module name such as ``"layer3.1"`` makes ``ts.slice`` trace
    through ``layer3``, so its residual blocks become nodes that can be cut between.
    The number of partitions is ``len(cuts) + 1``.
    """

    def __init__(self, cuts):
        if isinstance(cuts, (str, int)):
            cuts = [cuts]
        self.cuts = list(cuts)
        if not self.cuts:
            raise ValueError("NamedCutSplitter needs at least one cut")

    def expand_for(self, model) -> list:
        return [c.rpartition(".")[0] for c in self.cuts if isinstance(c, str) and "." in c]

    def split(self, model_graph, n_partitions: int | None = None) -> list:
        if n_partitions is not None and n_partitions != len(self.cuts) + 1:
            raise ValueError(f"{len(self.cuts)} cuts make {len(self.cuts) + 1} partitions, not n={n_partitions}")
        indices = [model_graph.index_of(c) for c in self.cuts]
        if indices != sorted(set(indices)):
            raise ValueError(f"cuts must be distinct and in model order; got {self.cuts} -> nodes {indices}")
        valid = model_graph.valid_cuts()
        for name, k in zip(self.cuts, indices):
            if k not in valid:
                before = [model_graph.nodes[v].name for v in valid if v < k][-1:]
                after = [model_graph.nodes[v].name for v in valid if v > k][:1]
                raise ValueError(
                    f"Can't cut after {name!r} (node {k}): more than one value would cross the cut "
                    f"(e.g. a skip connection spans it), or a later node reads the model input. "
                    f"Nearest valid cuts: before={before}, after={after}."
                    f"{self._cut_hint(model_graph)}"
                )
        return self.partitions_from_cuts(model_graph, indices)
