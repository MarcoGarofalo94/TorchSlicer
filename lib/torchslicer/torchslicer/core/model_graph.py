from dataclasses import dataclass, field
import copy
import operator

import torch
import torch.nn as nn


class TraceError(RuntimeError):
    """``torch.fx`` could not turn the model into a graph TorchSlicer can split faithfully."""


_PACK_HINT = (
    "\n\nTorchSlicer could not build the layer graph of this model automatically. Options:\n"
    "  1. Pass pack=: a function model -> list[nn.Module] whose stages each take and return one\n"
    "     tensor, e.g. ts.slice(model, n=4, pack=my_pack). Tracing is skipped entirely.\n"
    "     See docs/tutorials/pack.md and examples/pack/ (GPT-2, Llama, BERT, DeepSeek-MoE).\n"
    "  2. Make the top-level forward() traceable by torch.fx: submodules are leaves and are not\n"
    "     traced (unless listed in expand=), so moving data-dependent control flow or extra\n"
    "     inputs into a submodule is often enough."
)


def _human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def _tensors(x) -> list:
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (tuple, list)):
        return [t for v in x for t in _tensors(v)]
    if isinstance(x, dict):
        return [t for v in x.values() for t in _tensors(v)]
    return []


def _tensor_bytes(x) -> int:
    return sum(t.numel() * t.element_size() for t in _tensors(x))


def _shape_str(x) -> str:
    if isinstance(x, torch.Tensor):
        return str(list(x.shape))
    ts_ = _tensors(x)
    return f"{type(x).__name__}({len(ts_)} tensors)" if ts_ else type(x).__name__


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


class _Slot:
    """Placeholder for the i-th positional input inside a stored argument structure."""
    def __init__(self, i: int):
        self.i = i

    def __repr__(self):
        return f"in{self.i}"


def _fill(arg, inputs):
    """Replace every ``_Slot`` in a nested args structure with the matching input."""
    if isinstance(arg, _Slot):
        return inputs[arg.i]
    if isinstance(arg, tuple):
        return tuple(_fill(a, inputs) for a in arg)
    if isinstance(arg, list):
        return [_fill(a, inputs) for a in arg]
    if isinstance(arg, dict):
        return {k: _fill(v, inputs) for k, v in arg.items()}
    if isinstance(arg, slice):
        return slice(_fill(arg.start, inputs), _fill(arg.stop, inputs), _fill(arg.step, inputs))
    return arg


class _FunctionWrapper(nn.Module):
    """Any traced ``call_function`` / ``call_method`` node, e.g. ``F.relu(x)``,
    ``x.permute(0, 2, 1)``, ``torch.cat([a, b], 1)`` or ``x.view(x.size(0), -1)``.

    Graph-node arguments are stored as ``_Slot(i)`` and filled, in order, from the
    module's positional inputs; constant arguments are stored as they are. The target
    function is pickled by reference, so it must be importable on workers.
    """
    def __init__(self, kind: str, target, args: tuple, kwargs: dict):
        super().__init__()
        self.kind = kind          # "call_function" or "call_method"
        self.target = target      # callable, or method name for call_method
        self.args = args
        self.kwargs = kwargs

    def forward(self, *inputs):
        args = _fill(self.args, inputs)
        kwargs = _fill(self.kwargs, inputs)
        if self.kind == "call_method":
            return getattr(args[0], self.target)(*args[1:], **kwargs)
        return self.target(*args, **kwargs)

    def extra_repr(self):
        name = self.target if isinstance(self.target, str) else getattr(self.target, "__name__", repr(self.target))
        return f"{name}{self.args!r}" + (f", {self.kwargs!r}" if self.kwargs else "")


class _CallWrapper(nn.Module):
    """A submodule called with constant or keyword arguments besides its graph inputs."""
    def __init__(self, module: nn.Module, args: tuple, kwargs: dict):
        super().__init__()
        self.module = module
        self.args = args
        self.kwargs = kwargs

    def forward(self, *inputs):
        return self.module(*_fill(self.args, inputs), **_fill(self.kwargs, inputs))


class _AttrWrapper(nn.Module):
    """A parameter, buffer or constant the traced forward reads directly (``self.cls_token``).

    Parameters stay the same ``nn.Parameter`` object, so sharing with other nodes is
    seen as a tied weight. Takes no input (the activation passed in is ignored).
    """
    def __init__(self, value):
        super().__init__()
        if isinstance(value, nn.Parameter):
            self.value = value
        elif isinstance(value, torch.Tensor):
            self.register_buffer("value", value)
        else:
            self.value = value

    def forward(self, *_ignored):
        return self.value


# ── graph nodes ────────────────────────────────────────────────────────────────

@dataclass
class LayerNode:
    index: int
    module: nn.Module
    successors: list = field(default_factory=list)
    predecessors: list = field(default_factory=list)
    # Human-readable, unique name: the module's qualified name (``layer2.1``) or the
    # fx node name for functional ops (``flatten``, ``add_1``). Used by named cuts.
    name: str = ""
    # True if the node reads the model input directly; it must sit in the first partition.
    takes_input: bool = False
    # Enclosing module calls, outermost first (``["layer3", "layer3.0"]``; a module's
    # second call is ``"name:1"``). Lets a named cut refer to an expanded module.
    scope: list = field(default_factory=list)


# ── ModelGraph ─────────────────────────────────────────────────────────────────

class ModelGraph:
    def __init__(self, source: str = "unknown"):
        self.nodes: list[LayerNode] = []
        self.source = source
        # Submodules traced through (see from_module); recorded so the same graph can be rebuilt.
        self.expand: list | str = []
        # The model this graph was traced from (set by ``ts.trace``), for ``ts.slice(graph)``.
        self.model: nn.Module | None = None
        # ``ts.slice(..., multi_tensor=True)``: any gap may be cut (core/multicut.py packs it).
        self.multi_tensor = False

    # ── constructors ──────────────────────────────────────────────────────────

    @classmethod
    def _chain(cls, named_modules: list, source: str) -> "ModelGraph":
        graph = cls(source=source)
        n = len(named_modules)
        for i, (name, module) in enumerate(named_modules):
            graph.nodes.append(LayerNode(
                index=i,
                module=module,
                predecessors=[i - 1] if i > 0 else [],
                successors=[i + 1] if i < n - 1 else [],
                name=name,
                takes_input=(i == 0),
            ))
        return graph

    @classmethod
    def from_sequential(cls, network: nn.Module) -> "ModelGraph":
        """Build graph from network.children(), as a chain.

        Only correct when ``forward`` calls the children in registration order, each on the
        previous one's output, and does nothing else. Never used implicitly.
        """
        return cls._chain(list(network.named_children()), source="sequential")

    @classmethod
    def from_stages(cls, stages: list) -> "ModelGraph":
        """Build a sequential graph from a pre-packed list of nn.Module stages.

        Use this when you provide a ``pack`` function to ``ts.slice()`` — the
        returned list is treated as a flat sequential chain with no skip
        connections between stages. These stage boundaries are authoritative:
        recovery and re-slicing operate on the packed stages, not on the
        original model internals hidden inside them. Stage ``i`` is named ``str(i)``.
        """
        return cls._chain([(str(i), m) for i, m in enumerate(stages)], source="packed")

    @classmethod
    def from_module(cls, network: nn.Module, expand=None) -> "ModelGraph":
        """
        Build graph by tracing the module with torch.fx.

        By default the direct children of ``network`` are leaf modules (not traced
        into). ``expand`` lists submodules to trace through, so that their children
        become nodes instead, e.g. ``expand=["layer3"]`` turns a ResNet's ``layer3``
        into its residual blocks ``layer3.0``, ``layer3.1``. Ancestors of a listed
        name are expanded too. ``expand="all"`` traces everything down to the
        built-in ``torch.nn`` layers.

        Functional ops (``torch.flatten``, ``operator.add``, ``F.relu``, ``x.view``,
        …) become nodes wrapped as thin ``nn.Module``s, so every node is an ``nn.Module``.

        Raises ``TraceError`` (suggesting ``pack=``) if the model can't be traced, or if
        the traced graph can't be represented faithfully (several inputs or outputs,
        in-place ops whose result is unused, ...).
        """
        expand_set = cls._normalize_expand(network, expand)
        try:
            return cls._trace(network, expand_set)
        except TraceError:
            raise
        except Exception as e:
            raise TraceError(
                f"torch.fx could not trace {type(network).__name__}.forward: "
                f"{type(e).__name__}: {e}" + _PACK_HINT
            ) from e

    @staticmethod
    def _normalize_expand(network: nn.Module, expand) -> "set | str":
        if expand is None:
            return set()
        if expand == "all":
            return "all"
        if isinstance(expand, str):
            expand = [expand]
        known = {name for name, _ in network.named_modules() if name}
        result = set()
        for name in expand:
            if name not in known:
                raise ValueError(
                    f"expand: {type(network).__name__} has no submodule {name!r}. "
                    f"Top-level children: {[n for n, _ in network.named_children()]}"
                )
            parts = name.split(".")
            result.update(".".join(parts[:k]) for k in range(1, len(parts) + 1))
        return result

    @classmethod
    def _trace(cls, network: nn.Module, expand_set) -> "ModelGraph":
        import torch.fx as fx

        class _Tracer(fx.Tracer):
            """Direct children of the root (and of every expanded module) are leaves."""
            def is_leaf_module(self, m: nn.Module, qualified_name: str) -> bool:
                if expand_set == "all":
                    return super().is_leaf_module(m, qualified_name)
                if qualified_name in expand_set:
                    return False
                parent = qualified_name.rpartition(".")[0]
                if parent == "" or parent in expand_set:
                    return True
                return super().is_leaf_module(m, qualified_name)

        fx_graph = _Tracer().trace(network)
        gm = fx.GraphModule(network, fx_graph)
        fx_nodes = list(gm.graph.nodes)
        cls_name = type(network).__name__

        def node_args(n) -> list:
            """Distinct fx.Node arguments of n, in order of first appearance."""
            seen = []
            fx.node.map_arg((n.args, n.kwargs), lambda a: seen.append(a) if a not in seen else None)
            return seen

        # ── checks on the traced graph ───────────────────────────────────────
        placeholders = [n for n in fx_nodes if n.op == "placeholder"]
        output = next(n for n in fx_nodes if n.op == "output")
        out_val = output.args[0]
        if not isinstance(out_val, fx.Node):
            raise TraceError(
                f"{cls_name}.forward returns {type(out_val).__name__}, not a single tensor. "
                f"Each slice passes on exactly one tensor." + _PACK_HINT
            )

        # Keep only nodes the output depends on.
        live = {out_val}
        for n in reversed(fx_nodes):
            if n in live:
                live.update(node_args(n))
        for n in fx_nodes:
            if n in live or n.op in ("placeholder", "output"):
                continue
            target = n.target if isinstance(n.target, str) else getattr(n.target, "__name__", "")
            if target.endswith("_") and not target.endswith("__"):
                raise TraceError(
                    f"{cls_name}.forward calls the in-place op {target!r} (fx node {n.name!r}) "
                    f"and never uses its result; a graph cannot express that dependency." + _PACK_HINT
                )

        model_input = placeholders[0] if placeholders else None
        for p in placeholders[1:]:
            if p in live:
                raise TraceError(
                    f"{cls_name}.forward uses a second input {p.name!r}; a sliced model takes one "
                    f"input tensor. Extra inputs (masks, pixel_values, ...) can be passed to the first "
                    f"stage through an AuxInputStage in a pack= function." + _PACK_HINT
                )
        if out_val is model_input:
            raise TraceError(f"{cls_name}.forward returns its input unchanged; nothing to split.")

        input_users = [n for n in fx_nodes if n in live and model_input in node_args(n)]
        # The model input becomes an explicit node when it fans out, e.g. ``x + self.f(x)``.
        needs_input_node = len(input_users) > 1 or any(len(node_args(u)) > 1 for u in input_users)

        mg = cls(source="traced")
        mg.expand = sorted(expand_set) if isinstance(expand_set, set) else expand_set
        node_to_idx: dict = {}
        calls_seen: dict = {}

        if needs_input_node:
            node_to_idx[model_input] = 0
            mg.nodes.append(LayerNode(index=0, module=nn.Identity(), name="input", takes_input=True))

        for fx_node in fx_nodes:
            if fx_node not in live or fx_node.op in ("placeholder", "output"):
                continue
            inputs = node_args(fx_node)
            module = cls._node_to_module(fx_node, gm, inputs, node_to_idx)
            predecessors = cls._predecessors(fx_node, module, inputs, node_to_idx)

            if fx_node.op == "call_module":
                # first call: "layer1.0.relu"; later calls of the same module: "layer1.0.relu:1", ...
                k = calls_seen.get(fx_node.target, 0)
                calls_seen[fx_node.target] = k + 1
                name = fx_node.target if k == 0 else f"{fx_node.target}:{k}"
            elif fx_node.op == "get_attr":
                name = fx_node.target
            else:
                name = fx_node.name

            idx = len(mg.nodes)
            node_to_idx[fx_node] = idx
            mg.nodes.append(LayerNode(
                index=idx,
                module=module,
                predecessors=predecessors,
                successors=[],
                name=name,
                takes_input=model_input in inputs and not needs_input_node,
                scope=[k.replace("@", ":") for k in (fx_node.meta.get("nn_module_stack") or {})],
            ))

        for node in mg.nodes:
            for pred_idx in node.predecessors:
                mg.nodes[pred_idx].successors.append(node.index)
        return mg

    @staticmethod
    def _predecessors(fx_node, module, inputs, node_to_idx) -> list:
        """Graph indices feeding the node, in the order its module takes them."""
        if isinstance(module, _AddWrapper):
            return [node_to_idx[a] for a in fx_node.args]
        return [node_to_idx[a] for a in inputs if a in node_to_idx]

    @staticmethod
    def _node_to_module(fx_node, gm, inputs, node_to_idx) -> nn.Module:
        import torch.fx as fx

        def has_node(x) -> bool:
            found = []
            fx.node.map_arg(x, found.append)
            return bool(found)

        def slotted(x):
            # Graph inputs → _Slot(position in `inputs`); the model input, when it is
            # not a graph node, arrives as the single external input (slot 0).
            return fx.node.map_arg(x, lambda a: _Slot(inputs.index(a)))

        op, args, kwargs = fx_node.op, fx_node.args, fx_node.kwargs
        first_is_graph_node = bool(args) and isinstance(args[0], fx.Node)
        consts_only = first_is_graph_node and not has_node((args[1:], kwargs))

        if op == "get_attr":
            value = gm
            for part in fx_node.target.split("."):
                value = getattr(value, part)
            return _AttrWrapper(value)

        if op == "call_module":
            module = gm.get_submodule(fx_node.target)
            if not kwargs and list(args) == inputs:
                return module
            return _CallWrapper(module, slotted(args), slotted(kwargs))

        target = fx_node.target
        if op == "call_function":
            if target is torch.flatten and consts_only and not kwargs:
                start_dim = int(args[1]) if len(args) > 1 else 1
                end_dim = int(args[2]) if len(args) > 2 else -1
                return _FlattenWrapper(start_dim, end_dim)
            if (target is operator.add and len(args) == 2 and not kwargs
                    and all(isinstance(a, fx.Node) and a in node_to_idx for a in args)):
                return _AddWrapper()
        elif op == "call_method":
            if target == "flatten" and consts_only and not kwargs:
                start_dim = int(args[1]) if len(args) > 1 else 1
                end_dim = int(args[2]) if len(args) > 2 else -1
                return _FlattenWrapper(start_dim, end_dim)
            if target in ("view", "reshape") and consts_only and not kwargs and len(args) > 1:
                shape = args[1] if len(args) == 2 and isinstance(args[1], (tuple, list)) else args[1:]
                return _ReshapeWrapper(tuple(shape))

        return _FunctionWrapper(op, target, slotted(args), slotted(kwargs))

    # ── helpers ───────────────────────────────────────────────────────────────

    def is_dag(self) -> bool:
        """True if any node has more than one predecessor (not purely sequential)."""
        return any(len(n.predecessors) > 1 for n in self.nodes)

    def is_packed(self) -> bool:
        """True when the graph came from an explicit pack(model) stage list."""
        return self.source == "packed"

    def get_layers(self) -> list[nn.Module]:
        return [node.module for node in self.nodes]

    def names(self) -> list[str]:
        return [node.name for node in self.nodes]

    def index_of(self, name) -> int:
        """Node index for a node name (``"layer2"``), an index (``int``), or the name of an
        expanded module, which resolves to its output: the last node inside it."""
        if isinstance(name, int):
            if not -len(self.nodes) <= name < len(self.nodes):
                raise IndexError(f"node index {name} out of range for {len(self.nodes)} nodes")
            return name % len(self.nodes)
        for node in self.nodes:
            if node.name == name:
                return node.index
        inside = [node.index for node in self.nodes if name in node.scope]
        if inside:
            return inside[-1]
        raise KeyError(f"no graph node named {name!r}. Nodes: {self.names()}")

    def valid_cuts(self) -> list[int]:
        """Indices ``k`` such that cutting between node ``k`` and node ``k + 1`` is allowed:
        node ``k``'s output is the only value any later node needs from nodes ``<= k``,
        and no later node reads the model input. These are the only places where one
        tensor carries everything across the boundary.
        """
        n = len(self.nodes)
        if getattr(self, "multi_tensor", False):
            return list(range(n - 1))
        cuts = []
        max_consumer = -1         # highest index consuming the output of a node < k
        input_after = [False] * (n + 1)
        for i in range(n - 1, -1, -1):
            input_after[i] = input_after[i + 1] or self.nodes[i].takes_input
        for k in range(n - 1):
            if max_consumer <= k and not input_after[k + 1] and self.nodes[k].successors:
                cuts.append(k)
            succ = self.nodes[k].successors
            if succ:
                max_consumer = max(max_consumer, max(succ))
        return cuts

    def run_nodes(self, example_input, aux: dict | None = None) -> list:
        """Run the graph node by node on ``example_input`` and return every node's output
        (used by cost-based strategies to measure shapes, FLOPs and time)."""
        outputs = []
        for node in self.nodes:
            if node.predecessors:
                outputs.append(node.module(*[outputs[p] for p in node.predecessors]))
            elif aux and getattr(node.module, "accepts_aux_inputs", False):
                outputs.append(node.module(example_input, **aux))
            else:
                outputs.append(node.module(example_input))
        return outputs

    def summary(self, params: bool = False, example_input=None, aux: dict | None = None) -> str:
        """One line per node: index, name, type, predecessors, and whether a cut may follow it.

        ``params=True`` adds each node's parameter count and size (parameters + buffers);
        a tensor shared by several nodes (tied weights) is counted at its first node only.
        ``example_input`` adds each node's output shape and size, i.e. what crosses a cut
        placed right after it. It runs on a copy, so the model's weights, buffers and RNG
        state are not changed.
        """
        cuts = set(self.valid_cuts())
        sizes = self._param_sizes() if params else None
        outs = self._output_sizes(example_input, aux) if example_input is not None else None
        lines = [f"ModelGraph(source={self.source}, nodes={len(self)}, expand={self.expand or []})"]
        if sizes or outs:
            head = f"  {'#':>3}  {'name':<28} {'type':<20} "
            if sizes:
                head += f"{'params':>12} {'size':>10}  "
            if outs:
                head += f"{'output':<20} {'out size':>10}  "
            lines.append(head + "inputs")
        for node in self.nodes:
            mark = "  | cut ok" if node.index in cuts else ""
            line = f"  {node.index:>3}  {node.name:<28} {type(node.module).__name__:<20} "
            if sizes:
                count, nbytes, tied = sizes[node.index]
                line += f"{count:>12,} {_human_bytes(nbytes):>10}{'*' if tied else ' '} "
            if outs:
                shape, nbytes = outs[node.index]
                line += f"{shape:<20} {_human_bytes(nbytes):>10}  "
            lines.append(line + f"<- {node.predecessors}{mark}")
        if sizes:
            count = sum(s[0] for s in sizes)
            nbytes = sum(s[1] for s in sizes)
            lines.append(f"  total: {count:,} params, {_human_bytes(nbytes)} (parameters + buffers)")
            if any(s[2] for s in sizes):
                lines.append("  * shares tensors with an earlier node (tied weights), counted there")
        return "\n".join(lines)

    def view(self, example_input=None, aux: dict | None = None, title: str | None = None,
             multi_tensor: bool = False):
        """An interactive picture of the graph (``torchslicer.viz.SliceView``) where cuts are
        placed by clicking; it shows the matching ``ts.slice(...)`` call. ``multi_tensor``:
        every gap can be cut, labelled with what it would send (``ts.slice(multi_tensor=True)``)."""
        from ..viz import SliceView
        return SliceView(self, example_input, aux, title, multi_tensor)

    def _param_sizes(self) -> list:
        """Per node: (parameter count, bytes of parameters + buffers, shares a tensor with
        an earlier node). Each tensor is counted once, at the first node that holds it."""
        seen, sizes = set(), []
        for node in self.nodes:
            count = nbytes = 0
            tied = False
            for kind, tensors in (("p", node.module.parameters()), ("b", node.module.buffers())):
                for t in tensors:
                    if id(t) in seen:
                        tied = True
                        continue
                    seen.add(id(t))
                    nbytes += t.numel() * t.element_size()
                    if kind == "p":
                        count += t.numel()
            sizes.append((count, nbytes, tied))
        return sizes

    def _output_sizes(self, example_input, aux=None) -> list:
        """Per node: (output shape, output bytes), measured on a copy of the graph."""
        g = copy.copy(self)
        g.nodes = [copy.copy(n) for n in self.nodes]
        modules = copy.deepcopy([n.module for n in self.nodes])  # one memo: sharing kept
        for n, m in zip(g.nodes, modules):
            n.module = m
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            outputs = g.run_nodes(example_input, aux)
        return [(_shape_str(o), _tensor_bytes(o)) for o in outputs]

    def __len__(self) -> int:
        return len(self.nodes)
