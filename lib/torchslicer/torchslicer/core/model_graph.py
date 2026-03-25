from dataclasses import dataclass, field
import torch
import torch.nn as nn


# ── functional-op wrappers ─────────────────────────────────────────────────────

class _FlattenWrapper(nn.Module):
    """Wraps torch.flatten / tensor.flatten as nn.Module."""
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, self.start_dim, self.end_dim)

    def extra_repr(self):
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"


class _AddWrapper(nn.Module):
    """Wraps operator.add for residual / skip connections."""
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


class _ReshapeWrapper(nn.Module):
    """Wraps tensor.view / tensor.reshape."""
    def __init__(self, shape: tuple):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(self.shape)


# ── graph nodes ────────────────────────────────────────────────────────────────

@dataclass
class LayerNode:
    index: int
    module: nn.Module
    successors: list = field(default_factory=list)
    predecessors: list = field(default_factory=list)


# ── ModelGraph ─────────────────────────────────────────────────────────────────

class ModelGraph:
    def __init__(self):
        self.nodes: list[LayerNode] = []

    # ── constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_sequential(cls, network: nn.Module) -> "ModelGraph":
        """Build graph from network.children() — fast path for simple sequential models."""
        layers = list(network.children())
        graph = cls()
        for i, module in enumerate(layers):
            graph.nodes.append(LayerNode(
                index=i,
                module=module,
                predecessors=[i - 1] if i > 0 else [],
                successors=[i + 1] if i < len(layers) - 1 else [],
            ))
        return graph

    @classmethod
    def from_stages(cls, stages: list) -> "ModelGraph":
        """Build a sequential graph from a pre-packed list of nn.Module stages.

        Use this when you provide a ``pack`` function to ``ts.slice()`` — the
        returned list is treated as a flat sequential chain with no skip
        connections between stages.
        """
        graph = cls()
        n = len(stages)
        for i, module in enumerate(stages):
            graph.nodes.append(LayerNode(
                index=i,
                module=module,
                predecessors=[i - 1] if i > 0 else [],
                successors=[i + 1] if i < n - 1 else [],
            ))
        return graph

    @classmethod
    def from_module(cls, network: nn.Module) -> "ModelGraph":
        """
        Build graph by tracing the module with torch.fx.

        Treats direct children of ``network`` as leaf modules (does not recurse
        into them), and wraps functional ops (torch.flatten, operator.add, …)
        as thin nn.Module wrappers so every node is an nn.Module.

        Falls back to ``from_sequential()`` if tracing fails.
        """
        try:
            return cls._trace(network)
        except Exception:
            return cls.from_sequential(network)

    @classmethod
    def _trace(cls, network: nn.Module) -> "ModelGraph":
        import torch.fx as fx

        class _ShallowTracer(fx.Tracer):
            """Treats direct children of the root as leaves — no recursion into them."""
            def __init__(self, root: nn.Module):
                super().__init__()
                self._leaf_ids = {id(m) for m in root.children()}

            def is_leaf_module(self, m: nn.Module, qualified_name: str) -> bool:
                return id(m) in self._leaf_ids or super().is_leaf_module(m, qualified_name)

        tracer = _ShallowTracer(network)
        fx_graph = tracer.trace(network)
        gm = fx.GraphModule(network, fx_graph)

        mg = cls()
        node_to_idx: dict = {}

        for fx_node in gm.graph.nodes:
            if fx_node.op in ('placeholder', 'output', 'get_attr'):
                continue

            module = cls._node_to_module(fx_node, gm)
            if module is None:
                continue

            idx = len(mg.nodes)
            node_to_idx[fx_node] = idx

            predecessors = [
                node_to_idx[arg]
                for arg in fx_node.args
                if isinstance(arg, fx.Node) and arg in node_to_idx
            ]

            mg.nodes.append(LayerNode(
                index=idx,
                module=module,
                predecessors=predecessors,
                successors=[],
            ))

        # Fill successors
        for node in mg.nodes:
            for pred_idx in node.predecessors:
                mg.nodes[pred_idx].successors.append(node.index)

        return mg

    @staticmethod
    def _node_to_module(fx_node, gm) -> nn.Module | None:
        import torch.fx as fx
        import operator

        if fx_node.op == 'call_module':
            return gm.get_submodule(fx_node.target)

        if fx_node.op == 'call_function':
            fn = fx_node.target
            args = fx_node.args

            if fn is torch.flatten:
                start_dim = int(args[1]) if len(args) > 1 else 1
                end_dim = int(args[2]) if len(args) > 2 else -1
                return _FlattenWrapper(start_dim, end_dim)

            if fn is operator.add:
                return _AddWrapper()

            # Unrecognised function — skip (don't add as a node)
            return None

        if fx_node.op == 'call_method':
            method = fx_node.target
            args = fx_node.args
            if method == 'flatten':
                start_dim = int(args[1]) if len(args) > 1 else 1
                end_dim = int(args[2]) if len(args) > 2 else -1
                return _FlattenWrapper(start_dim, end_dim)
            if method in ('view', 'reshape'):
                shape = tuple(a for a in args[1:] if not isinstance(a, type(fx_node)))
                return _ReshapeWrapper(shape)

        return None

    # ── helpers ───────────────────────────────────────────────────────────────

    def is_dag(self) -> bool:
        """True if any node has more than one predecessor (not purely sequential)."""
        return any(len(n.predecessors) > 1 for n in self.nodes)

    def get_layers(self) -> list[nn.Module]:
        return [node.module for node in self.nodes]

    def __len__(self) -> int:
        return len(self.nodes)
