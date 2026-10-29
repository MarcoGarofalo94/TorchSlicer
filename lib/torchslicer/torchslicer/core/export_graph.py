"""Build a ``ModelGraph`` with ``torch.export`` (``ts.slice(..., tracer="export")``).

``torch.fx`` traces ``forward`` symbolically and fails on many real models (Hugging Face
transformers: keyword-heavy signatures, decorators, cache objects). ``torch.export``
(non-strict) runs ``forward`` on fake tensors instead, so it needs an example input, and
gives a graph of ATen ops with symbolic sizes. Each op becomes a node (``_OpWrapper``),
parameters and buffers become ``_AttrWrapper`` nodes placed right before the node that
reads them, and the module each op came from is kept in ``LayerNode.scope``, so named cuts
(``cuts=["transformer.h.1"]``) work as with ``expand="all"``.

The graph is the model's training-mode forward. What depends on the mode at run time is
handled per op: dropout (also inside ``scaled_dot_product_attention``), batch norm, and
running-statistics updates follow the slice's ``train()``/``eval()``. Other Python
branches on ``self.training`` are fixed at their training value.
"""
from __future__ import annotations

import operator

import torch
import torch.nn as nn

from .model_graph import (LayerNode, ModelGraph, TraceError, _AttrWrapper, _FunctionWrapper,
                          _PACK_HINT, _Slot, _fill)

# ops whose ``train`` flag follows the slice's mode: {op name: position of the flag}
_TRAIN_FLAG = {
    "aten::dropout": 2, "aten::feature_dropout": 2, "aten::alpha_dropout": 2,
    "aten::feature_alpha_dropout": 2, "aten::native_dropout": 2,
    "aten::batch_norm": 5, "aten::_native_batch_norm_legit": 5,
}
# ops that update running statistics in their buffer arguments
_STATS_OPS = {"aten::batch_norm", "aten::_native_batch_norm_legit", "aten::native_batch_norm",
              "aten::instance_norm", "aten::cudnn_batch_norm"}
_DROPPED = {"aten::_assert_tensor_metadata", "aten::_assert_scalar", "aten::sym_constrain_range_for_size",
            "aten::_assert_async", "aten::sym_constrain_range"}


def _is_view(op) -> bool:
    """An ATen op whose output shares its input's memory (``slice``, ``transpose``, ...)."""
    if not isinstance(op, torch._ops.OpOverload) or not op._schema.returns:
        return False
    alias = op._schema.returns[0].alias_info
    return alias is not None and not alias.is_write


def _op_name(op) -> str:
    return op._schema.name if isinstance(op, torch._ops.OpOverload) else ""


def _resolve_op(qualname: str):
    """``"aten::relu_.default"`` -> ``torch.ops.aten.relu_.default``."""
    ns, rest = qualname.split("::", 1)
    name, _, overload = rest.partition(".")
    return getattr(getattr(getattr(torch.ops, ns), name), overload or "default")


class _OpWrapper(_FunctionWrapper):
    """One ATen op of an exported graph. Pickled by the op's name (``OpOverload`` objects
    don't pickle). A ``device=`` argument follows the slice's device (the graph records
    the device it was exported on); a ``train`` flag follows ``self.training``."""

    def __init__(self, op, args: tuple, kwargs: dict, side_effect: bool = False):
        super().__init__("call_function", op, args, kwargs)
        name = _op_name(op)
        self.train_arg = _TRAIN_FLAG.get(name)
        self.sdpa = name == "aten::scaled_dot_product_attention"
        # an update of module state whose result is unused (``num_batches_tracked += 1``):
        # training-mode only, like the module code it came from
        self.side_effect = side_effect
        if any(isinstance(v, torch.device) for v in kwargs.values()):
            self.register_buffer("_device", torch.empty(0), persistent=False)

    @property
    def op_name(self) -> str:
        """``"relu_"`` for ``aten.relu_.default``."""
        return self.target.overloadpacket.__name__

    def forward(self, *inputs):
        if self.side_effect and not self.training:
            return None
        args = list(_fill(self.args, inputs))
        kwargs = _fill(self.kwargs, inputs)
        if "_device" in self._buffers:
            kwargs = {k: (self._device.device if isinstance(v, torch.device) else v)
                      for k, v in kwargs.items()}
        if self.train_arg is not None and len(args) > self.train_arg:
            args[self.train_arg] = args[self.train_arg] and self.training
        if self.sdpa and not self.training:
            if len(args) > 4:
                args[4] = 0.0
            if "dropout_p" in kwargs:
                kwargs = {**kwargs, "dropout_p": 0.0}
        return self.target(*args, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["target"] = str(self.target._schema.name) + "." + (self.target._overloadname or "default")
        return state

    def __setstate__(self, state):
        state = dict(state)
        state["target"] = _resolve_op(state["target"])
        self.__dict__.update(state)

    def extra_repr(self):
        return f"{self.target}{self.args!r}" + (f", {self.kwargs!r}" if self.kwargs else "")


class _InputGuard(nn.Module):
    """The model input, checked against what the export assumed: sizes it fixed must match,
    sizes it kept symbolic must lie in their range (a size of 1 is allowed where the
    export assumed >= 2, as PyTorch's export does by convention)."""

    def __init__(self, dims: list):
        super().__init__()
        self.dims = dims            # per dim: int (fixed), or (lo, hi | None)

    def forward(self, x):
        return _check_input(x, self.dims, "input")

    def extra_repr(self):
        return repr(self.dims)


def _check_input(x, dims, what, *others):
    """``x`` (a tensor, or a size) checked against what an export assumed: per dim, a fixed
    size, a range ``(lo, hi | None)`` (a size of 1 is allowed where the export assumed >= 2,
    as PyTorch's export does by convention), or ``("=", j, d)``: equal to dim ``d`` of
    ``others[j]`` (``d`` None: ``others[j]`` itself, a size)."""
    is_tensor = isinstance(x, torch.Tensor)
    if (is_tensor and x.dim() != len(dims)) or (not is_tensor and not isinstance(x, int)):
        raise ValueError(f"the graph was exported for a {len(dims)}-d {what}, got "
                         f"{tuple(x.shape) if is_tensor else type(x).__name__}")
    for d, (size, spec) in enumerate(zip(x.shape if is_tensor else [x], dims)):
        if isinstance(spec, int):
            ok, want = size == spec, f"size {spec}"
        elif spec[0] == "=":
            ref = others[spec[1]] if spec[2] is None else others[spec[1]].shape[spec[2]]
            ok, want = size == ref, f"a size equal to another input's ({ref})"
        else:
            lo, hi = spec
            ok, want = size >= (1 if lo <= 2 else lo) and (hi is None or size <= hi), f"sizes in [{lo}, {hi}]"
        if not ok:
            raise ValueError(f"{what}{f' dim {d}' if is_tensor else ''} is {size}, but the graph was "
                             f"exported with {want}: trace again with an example_input of that shape")
    return x


def _output_key(out, cls_name: str):
    if isinstance(out, torch.Tensor):
        return None
    if hasattr(out, "keys"):
        keys = [k for k in out.keys() if isinstance(out[k], torch.Tensor)]
        if "logits" in keys:
            return "logits"
        if len(keys) == 1:
            return keys[0]
        what = f"{type(out).__name__} with tensors {keys}"
    elif isinstance(out, (tuple, list)):
        idx = [i for i, v in enumerate(out) if isinstance(v, torch.Tensor)]
        if len(idx) == 1:
            return idx[0]
        what = f"{type(out).__name__} of {len(out)} values"
    else:
        what = type(out).__name__
    raise TraceError(f"{cls_name}.forward returns a {what}; a sliced model ends in one tensor. "
                     f"Wrap the model in a module whose forward returns the one you train on.")


def _is_sym(n) -> bool:
    return isinstance(n.meta.get("val"), (torch.SymInt, torch.SymFloat, torch.SymBool))


def _localize_sizes(graph, max_visit: int = 64):
    """``torch.export`` reads each symbolic size once, from the input, at the top of the
    graph, and every reshape deep in the model refers to it: one value that would cross
    every cut. Read it again, next to the node that needs it, from the nearest tensor
    that has it as a dimension (or rebuild an expression of such sizes)."""
    import torch.fx as fx

    def tensors_up(user):
        """Tensor-valued ancestors of ``user``, nearest first."""
        seen, queue = set(), [a for a in user.all_input_nodes]
        while queue and len(seen) < max_visit:
            n = queue.pop(0)
            if n in seen:
                continue
            seen.add(n)
            if isinstance(n.meta.get("val"), torch.Tensor) and n.op != "get_attr":
                yield n
            queue.extend(n.all_input_nodes)

    def local(user, sym, depth=0):
        expr = sym.meta["val"].node.expr
        for t in tensors_up(user):
            for d, size in enumerate(t.meta["val"].shape):
                if isinstance(size, torch.SymInt) and size.node.expr == expr:
                    if sym.target is torch.ops.aten.sym_size.int and tuple(sym.args) == (t, d):
                        return sym
                    with graph.inserting_before(user):
                        new = graph.call_function(torch.ops.aten.sym_size.int, (t, d))
                    new.meta["val"] = sym.meta["val"]
                    return new
        if depth < 4 and sym.op == "call_function" and not isinstance(sym.target, torch._ops.OpOverload):
            args = [local(user, a, depth + 1) if isinstance(a, fx.Node) and _is_sym(a) else a
                    for a in sym.args]
            if any(a is None for a in args):
                return None
            with graph.inserting_before(user):
                new = graph.call_function(sym.target, tuple(args), sym.kwargs)
            new.meta["val"] = sym.meta["val"]
            return new
        return None

    for user in list(graph.nodes):
        if user.op != "call_function" or _is_sym(user):
            continue
        cache = {}

        def repl(a):
            if not _is_sym(a) or a.op != "call_function":
                return a
            if a not in cache:
                cache[a] = local(user, a) or a
            return cache[a]

        user.args = fx.node.map_arg(user.args, repl)
        user.kwargs = fx.node.map_arg(user.kwargs, repl)


def export_graph(network: nn.Module, example_input, prefer: str = "export", expand="all",
                 mix: bool = True) -> ModelGraph:
    """Trace ``network`` on ``example_input`` (one batch's inputs, core/inputs.py; its batch
    size must be >= 2: export fixes sizes of 1) into a ``ModelGraph``: with ``torch.export``
    (one node per ATen op), and ``torch.fx`` for the modules export can't trace
    (core/hybrid.py). ``prefer="fx"``: fx first, export for the modules fx can't open
    (``expand``: which ones to open). ``mix=False``: the ``prefer``red tracer only."""
    from .hybrid import Builder
    from .inputs import bind, input_leaves, main_name, split_inputs

    cls_name = type(network).__name__
    leaves = input_leaves(example_input)
    if not leaves:
        raise TraceError(f"example_input holds no tensor: {type(example_input).__name__}")
    main = split_inputs(example_input)[0]
    if main.dim() and main.shape[0] < 2:
        raise ValueError("tracing on an example_input needs a batch of at least 2 "
                         "(torch.export fixes a size of 1, so other batch sizes would fail)")
    args, kwargs = bind(network, example_input)
    was_training = network.training
    try:
        network.eval()
        with torch.no_grad():
            key = _output_key(network(*args, **kwargs), cls_name)
    finally:
        network.train(was_training)
    builder = Builder(network, prefer, expand, mix)
    names = [n for n, _ in leaves]
    graph = builder.build((args, kwargs), key, names if len(leaves) > 1 else None)
    if builder.input_dims is not None:
        guard = _InputGuard(builder.input_dims)
    else:
        guard = nn.Identity()
    source = ("exported" if builder.kinds == {"export"} else
              "mixed" if "export" in builder.kinds else "traced")
    inputs = None
    if len(leaves) > 1:
        inputs = [(n, n == main_name(names), _bytes(t)) for n, t in leaves]
    mg = convert(graph, network, builder.consts, guard, _bytes(main),
                 source=source, expand=expand, inputs=inputs)
    if any(n.out_bytes is None for n in mg.nodes):        # nodes fx made: measured on the input
        modes = [(m, m.training) for m in network.modules()]
        cuda = sorted({t.device for _, t in leaves if t.is_cuda}, key=str)
        try:
            network.eval()
            for node in mg.nodes:
                node.module.eval()
            with torch.no_grad(), torch.random.fork_rng(devices=cuda):
                outs = mg.run_nodes(example_input)
        finally:
            for node in mg.nodes:
                node.module.train()
            for m, mode in modes:
                m.training = mode
        for node, out in zip(mg.nodes, outs):
            if node.out_bytes is None:
                node.out_bytes = _bytes(out)
    return mg


def convert(graph, network: nn.Module, consts: dict, input_module: nn.Module, input_bytes: float,
            source: str = "exported", expand="all", inputs: list | None = None) -> ModelGraph:
    """The ``ModelGraph`` of a ``torch.fx`` graph whose nodes come from ``torch.export``
    (ATen ops, ``meta["ts_kind"] == "export"``), from ``torch.fx`` (``"fx"``), or are calls
    of modules neither could open (``"leaf"``); ``meta["ts_scope"]``: the module calls each
    node is inside. get_attr targets are paths in ``network``, or keys of ``consts``.
    ``inputs``: a model with several inputs, ``[(name, is main, bytes)]`` per placeholder
    (one ``input.<name>`` node each, for those it uses); else ``input_module`` is the input."""
    import torch.fx as fx
    from .inputs import _Input

    cls_name = type(network).__name__
    nodes = list(graph.nodes)
    placeholders = [n for n in nodes if n.op == "placeholder"]
    placeholder = placeholders[0]
    output = next(n for n in nodes if n.op == "output")
    out_val = output.args[0]
    if out_val in placeholders:
        raise TraceError(f"{cls_name}.forward returns its input unchanged; nothing to split.")

    def kind(n) -> str:
        return n.meta.get("ts_kind", "export")

    def attr_value(target: str):
        if target in consts:
            return consts[target]
        for get in (network.get_parameter, network.get_buffer):
            try:
                return get(target)
            except (AttributeError, KeyError, ValueError):
                pass
        value = network
        for part in target.split("."):
            value = getattr(value, part)
        return value

    def is_op(n, names) -> bool:
        return n.op == "call_function" and _op_name(n.target) in names

    def mutates(n, a) -> bool:
        """Does op node ``n`` write its argument ``a`` (in place, or as running stats)?"""
        if n.op != "call_function" or not isinstance(n.target, torch._ops.OpOverload):
            return False
        if _op_name(n.target) in _STATS_OPS:
            return a in n.args[3:5]
        schema = n.target._schema
        return any(arg.alias_info is not None and arg.alias_info.is_write and i < len(n.args)
                   and n.args[i] is a for i, arg in enumerate(schema.arguments))

    def base_of(a):
        """The tensor whose memory ``a`` is (``a`` itself, or what ``a`` is a view of)."""
        while a.op == "call_function" and (a.target is operator.getitem or _is_view(a.target)):
            a = a.args[0]
        return a

    # in-place writes: {op node: the tensors whose memory it writes}
    writes = {n: {base_of(a) for a in n.all_input_nodes if mutates(n, a)} for n in nodes}
    writes = {n: bases for n, bases in writes.items() if bases}
    stateful = {b for bases in writes.values() for b in bases if b.op == "get_attr"}

    # ── which nodes are kept ────────────────────────────────────────────────
    # an update of module state whose result is unused (``num_batches_tracked += 1``)
    side_effects = [n for n, bases in writes.items() if not n.users and bases <= stateful]
    live = {out_val, *side_effects} | ({placeholder} if inputs is None else set())
    while True:
        for n in reversed(nodes):
            if n in live:
                live.update(n.all_input_nodes)
        # a write into a tensor that is read later (``mask[i:j] = 1``, ``x.masked_fill_(...)``)
        more = [n for n, bases in writes.items() if n not in live and bases & live]
        if not more:
            break
        live.update(more)
    live = {n for n in live if not is_op(n, _DROPPED)}
    in_place = [(b, n) for n, bases in writes.items() if n in live for b in bases if b not in stateful]

    # ── build the ModelGraph ────────────────────────────────────────────────
    mg = ModelGraph(source=source)
    mg.expand = expand if isinstance(expand, str) else sorted(expand)
    if inputs is None:
        mg.nodes.append(LayerNode(index=0, module=input_module, name="input", takes_input=True,
                                  out_bytes=input_bytes))
        node_to_idx = {placeholder: 0}
    else:            # one node per input the model uses; no cut before the last of them
        used = [(p, spec) for p, spec in zip(placeholders, inputs) if p in live]
        if not used:
            raise TraceError(f"{cls_name}.forward uses none of its inputs; nothing to split.")
        mg.inputs = [name for _, (name, _, _) in used]
        node_to_idx = {}
        for p, (name, main, nbytes) in used:
            node_to_idx[p] = len(mg.nodes)
            mg.nodes.append(LayerNode(index=len(mg.nodes), module=_Input(name, main),
                                      name=f"input.{name}", takes_input=True, out_bytes=nbytes))
    uses: dict = {}
    names = {node.name for node in mg.nodes}

    def unique(name: str) -> str:
        while name in names:
            name += "'"
        names.add(name)
        return name

    def numbered(target: str) -> str:
        """``target``, then ``target:1``, ... for later uses of the same module or attribute."""
        k = uses.get(target, 0)
        uses[target] = k + 1
        return unique(target + (f":{k}" if k else ""))

    def add(module, preds, name, scope=(), out_bytes=None):
        idx = len(mg.nodes)
        mg.nodes.append(LayerNode(index=idx, module=module, predecessors=preds, name=name,
                                  scope=list(scope), out_bytes=out_bytes))
        return idx

    def attr_node(a, user):
        if a in stateful and a in node_to_idx:
            return node_to_idx[a]
        value = attr_value(a.target)
        idx = add(_AttrWrapper(value), [], numbered(a.target), user.meta.get("ts_scope", []), _bytes(value))
        if a in stateful:
            mg.nodes[idx].module.stateful = True
            node_to_idx[a] = idx
        return idx

    for n in nodes:
        if n not in live or n.op in ("placeholder", "output"):
            continue
        if n.op == "get_attr" and kind(n) == "export":
            continue                       # read right before each user (attr_node)
        inputs = []
        for a in n.all_input_nodes:
            if a not in inputs:
                inputs.append(a)
        local = {}
        for a in inputs:
            local[a] = attr_node(a, n) if a.op == "get_attr" and kind(a) == "export" else node_to_idx[a]
        slotted = lambda x: fx.node.map_arg(x, lambda a: _Slot(inputs.index(a)))
        scope = n.meta.get("ts_scope", [])
        if kind(n) == "export":
            if isinstance(n.target, torch._ops.OpOverload):
                # ``x.contiguous().view(...)`` is traced as a bare view when the export's kernels
                # said ``x`` was contiguous; at run time another kernel (another attention
                # backend) may lay ``x`` out differently. reshape is the same view where view
                # works, and a copy where it would fail.
                op = torch.ops.aten.reshape.default if n.target is torch.ops.aten.view.default else n.target
                module = _OpWrapper(op, slotted(n.args), slotted(n.kwargs), side_effect=n in side_effects)
            else:
                module = _FunctionWrapper("call_function", n.target, slotted(n.args), slotted(n.kwargs))
            preds, name = [local[a] for a in inputs], unique(n.name)
        else:
            if n.op == "get_attr":
                module = _AttrWrapper(attr_value(n.target))
            elif "ts_check" in n.meta:
                from .hybrid import _OpaqueCall
                module = _OpaqueCall(network.get_submodule(n.target), slotted(n.args), slotted(n.kwargs),
                                     n.meta["ts_check"])
            else:
                module = ModelGraph._node_to_module(n, network, inputs, local)
            preds = ModelGraph._predecessors(n, module, inputs, local)
            name = numbered(n.target) if n.op in ("call_module", "get_attr") else unique(n.name)
        out_bytes = _bytes(n.meta["val"]) if kind(n) == "export" and "val" in n.meta else None
        node_to_idx[n] = add(module, preds, name, scope, out_bytes)

    # A tensor written in place stays where it is made: rebuilt after a cut it would miss the
    # write, and sent across a cut before the write, the write would change only a copy.
    written = [(mg.nodes[node_to_idx[b]], mg.nodes[node_to_idx[n]]) for b, n in in_place]
    for base, _ in written:
        base.module.mutated = True
    out_idx = node_to_idx[out_val]
    if out_idx != len(mg.nodes) - 1:          # side effects traced after the output: move them up
        order = [i for i in range(len(mg.nodes)) if i != out_idx] + [out_idx]
        _reorder(mg, order)
    mg.no_cut = {k for base, write in written for k in range(base.index, write.index)}
    mg.no_cut |= set(range(len(mg.inputs or []) - 1))
    for node in mg.nodes:
        for p in node.predecessors:
            mg.nodes[p].successors.append(node.index)
    for n in nodes:            # a structured value (a module's output tuple, ModelOutput):
        if n.meta.get("ts_struct") and n in node_to_idx:      # no cut before its items are read
            node = mg.nodes[node_to_idx[n]]
            mg.no_cut |= set(range(node.index, max(node.successors, default=node.index)))
    return mg


def _hint(size) -> float:
    if isinstance(size, int):
        return size
    hint = size.node.hint
    if hint is None:
        hint = size.node.shape_env.size_hint(size.node.expr, allow_none=True)
    return float("inf") if hint is None else int(hint)


def _bytes(val) -> float:
    """Bytes of a traced value on the example input (from its symbolic shape)."""
    if isinstance(val, torch.Tensor):
        n = 1
        for size in val.shape:
            n *= _hint(size)
        return n * val.element_size()
    if isinstance(val, (tuple, list)):
        return sum(_bytes(v) for v in val)
    if isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool, int, float, bool)):
        return 8
    return 0


def _input_dims(placeholder, ep, seen=None) -> list:
    """What the export assumed about an input's sizes (``_check_input``). ``seen``
    ({symbol: (input, dim)}, filled as it goes): a size already seen at an earlier input is
    checked equal to it."""
    val = placeholder.meta["val"]
    dims = []
    for d, size in enumerate(val.shape if isinstance(val, torch.Tensor) else [val]):
        if isinstance(size, torch.SymInt):
            expr = size.node.expr
            if seen is not None and expr in seen:
                dims.append(("=", *seen[expr]))
                continue
            if seen is not None:
                seen[expr] = (placeholder, d if isinstance(val, torch.Tensor) else None)
            vr = ep.range_constraints.get(expr)
            lo = int(vr.lower) if vr is not None else 2
            hi = vr.upper if vr is not None else None
            hi = None if hi is None or not getattr(hi, "is_finite", True) or str(hi) == "int_oo" else int(hi)
            dims.append((lo, hi))
        else:
            dims.append(int(size))
    return dims


def _reorder(mg, order):
    new_index = {old: new for new, old in enumerate(order)}
    nodes = [mg.nodes[i] for i in order]
    for node in nodes:
        node.index = new_index[node.index]
        node.predecessors = [new_index[p] for p in node.predecessors]
        node.successors = []
    for node in nodes:
        if any(p >= node.index for p in node.predecessors):
            raise TraceError("the exported graph has a side effect after its output that depends on it")
    mg.nodes = nodes


_SIZE_READS = {"aten::sym_size", "aten::sym_numel", "aten::sym_stride", "aten::sym_storage_offset"}


def is_size_read(module) -> bool:
    """``aten.sym_size.int`` and friends: a tensor's metadata, as ``x.size(1)`` under fx."""
    return isinstance(module, _OpWrapper) and _op_name(module.target) in _SIZE_READS


__all__ = ["export_graph", "convert", "is_size_read", "_OpWrapper", "_InputGuard"]
