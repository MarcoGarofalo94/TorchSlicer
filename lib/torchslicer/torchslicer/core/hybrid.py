"""Trace a model with ``torch.fx`` and ``torch.export`` together, module by module.

Used by ``tracer="export"``, and by ``tracer="auto"`` when fx alone fails (both need
``example_input=``). Each module is opened by the preferred tracer (export for
``"export"``, fx for ``"auto"``); if its own code defeats that tracer, by the other one;
if that fails too, it stays one node that calls it.

- fx opens a module down to the leaves ``expand`` asks for. A submodule whose own code fx
  can't trace (``if x.shape[1] > 8:``, ``**kwargs`` passed on, ...) is opened with export,
  on the inputs it gets in an eager run.
- export opens everything below. A submodule export can't trace (a check on a tensor's
  values, an op without a fake kernel) is replaced during the export by an opaque op
  (``torchslicer::opaque``) whose output shapes follow its inputs as in an eager run, and
  is opened on its own with fx.

Which module defeated a tracer is read from the traceback (the innermost frame of a
module's code). The pieces are inlined, in execution order, into one ``torch.fx`` graph,
which ``export_graph.convert`` turns into a ``ModelGraph``.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import operator
import sys

import torch
import torch.fx as fx
import torch.nn as nn
from torch.utils import _pytree as pytree

from .export_graph import _check_input
from .model_graph import TraceError, _CallWrapper, _fill, _traced_factories


# ── the opaque op: a module call export can't trace ────────────────────────────

_OPAQUE: dict = {}     # key -> _Opaque, for the export in progress


@torch.library.custom_op("torchslicer::opaque", mutates_args=())
def _opaque(key: int, tensors: list[torch.Tensor], ints: list[int]) -> list[torch.Tensor]:
    raise RuntimeError("torchslicer::opaque stands for a module call while tracing; it never runs")


@_opaque.register_fake
def _(key, tensors, ints):
    return _OPAQUE[key].fake(tensors, ints)


def _shape_rules(leaves: list, dynamic: list, outs: list) -> list:
    """How each output tensor's shape follows the call's inputs, from one eager call:
    per output (dtype, dims), each dim ``(pos, d)`` (dim d of the input leaf at ``pos``),
    ``(pos,)`` (the int input at ``pos``) or a constant."""
    rules = []
    for out in outs:
        dims = []
        for d, size in enumerate(out.shape):
            same = [(p, d) for p in dynamic if isinstance(leaves[p], torch.Tensor)
                    and leaves[p].dim() > d and leaves[p].shape[d] == size]
            other = [(p, e) for p in dynamic if isinstance(leaves[p], torch.Tensor)
                     for e, s in enumerate(leaves[p].shape) if s == size]
            ints = [(p,) for p in dynamic if not isinstance(leaves[p], torch.Tensor) and leaves[p] == size]
            dims.append(int(size) if size == 1 else (same + other + ints + [int(size)])[0])
        rules.append((out.dtype, dims))
    return rules


def _size(rule, leaves):
    if isinstance(rule, int):
        return rule
    return leaves[rule[0]].shape[rule[1]] if len(rule) == 2 else leaves[rule[0]]


class _Opaque:
    """One call of a module export can't trace: what the opaque op stands for."""

    def __init__(self, path, k, spec, leaves, tensors, ints, out_spec, out_leaves, rules, device):
        self.path, self.k = path, k
        self.spec, self.leaves = spec, leaves          # the call's (args, kwargs), dynamic leaves None
        self.tensors, self.ints = tensors, ints        # positions of the dynamic leaves
        self.out_spec, self.out_leaves = out_spec, out_leaves
        self.rules, self.device = rules, device

    def fill(self, tensors, ints) -> list:
        leaves = list(self.leaves)
        for p, v in zip(self.tensors, tensors):
            leaves[p] = v
        for p, v in zip(self.ints, ints):
            leaves[p] = v
        return leaves

    def fake(self, tensors, ints):
        leaves = self.fill(tensors, ints)
        device = tensors[0].device if tensors else self.device
        return [torch.empty([_size(r, leaves) for r in dims], dtype=dtype, device=device)
                for dtype, dims in self.rules]


class _OpaqueCall(_CallWrapper):
    """A module call kept whole inside an exported graph. The ops after it were traced
    with output shapes inferred from one eager call; a call that doesn't follow them fails
    here, not later in an op that assumed them."""

    def __init__(self, module, args, kwargs, rules):
        super().__init__(module, args, kwargs)
        self.rules = rules

    def forward(self, *inputs):
        args, kwargs = _fill(self.args, inputs), _fill(self.kwargs, inputs)
        out = self.module(*args, **kwargs)
        leaves = pytree.tree_leaves((args, kwargs))
        outs = [o for o in pytree.tree_leaves(out) if isinstance(o, torch.Tensor)]
        for o, (_, dims) in zip(outs, self.rules):
            want = [_size(r, leaves) for r in dims]
            if list(o.shape) != want:
                raise RuntimeError(
                    f"{type(self.module).__name__} returned shape {tuple(o.shape)}, but the graph "
                    f"was traced assuming {tuple(want)} (inferred from one eager call, since "
                    f"torch.export couldn't trace this module): trace again with an "
                    f"example_input of the shape you run")
        return out


# ── helpers ────────────────────────────────────────────────────────────────────

class _Untraceable(Exception):
    """A tracer failed on a module's own code."""


class _Struct:
    """A structured value (tuple, dict, ``ModelOutput``) whose leaves are graph values:
    ``out[0]`` and ``out.logits`` resolve to the leaf without a node."""

    def __init__(self, spec, flat):
        self.spec, self.flat = spec, flat

    def get(self, key):
        spec, n = self.spec, self.spec.num_children
        if isinstance(key, int) and (spec.type in (tuple, list) or hasattr(spec.type, "to_tuple")):
            i = key + n if key < 0 else key
        elif isinstance(key, str) and isinstance(spec.context, (list, tuple)) and key in spec.context:
            i = list(spec.context).index(key)
        else:
            return _MISSING
        if not 0 <= i < n:
            return _MISSING
        start = sum(spec.child(j).num_leaves for j in range(i))
        child = spec.child(i)
        flat = self.flat[start:start + child.num_leaves]
        return flat[0] if child.is_leaf() else _Struct(child, flat)


_MISSING = object()


def _rebuild(spec, *leaves):
    return pytree.tree_unflatten(list(leaves), spec)


def _leaf_item(out, i):
    """The i-th leaf of a structured module output."""
    return pytree.tree_leaves(out)[i]


def _leaves(call) -> list:
    """Leaves of a call's (args, kwargs) of graph values, a ``_Struct`` giving its own."""
    out = []
    for v in pytree.tree_leaves(call, is_leaf=lambda v: isinstance(v, (_Struct, fx.Node))):
        out.extend(v.flat if isinstance(v, _Struct) else [v])
    return out


def _chain(error) -> list:
    out = []
    while error is not None and error not in out:
        out.append(error)
        error = error.__cause__ or error.__context__
    return out


def _culprit(error, modules: dict, outer=()):
    """The innermost module of ``modules`` ({id: path}) whose code was running when
    ``error`` was raised, looking through chained exceptions but not ``outer`` (those
    being handled when tracing started: Python chains them to every new exception)."""
    found, seen = None, {id(e) for e in outer}
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        tb = error.__traceback__
        while tb is not None:
            owner = tb.tb_frame.f_locals.get("self")
            if id(owner) in modules:
                found = modules[id(owner)]
            tb = tb.tb_next
        error = error.__cause__ or error.__context__
    return found


def _short(error) -> str:
    text = str(error).strip()
    return f"{type(error).__name__}: {text.splitlines()[0] if text else ''}"


def _join(path: str, rel: str) -> str:
    return f"{path}.{rel}" if path and rel else path or rel


def _devices(call) -> list:
    return sorted({t.device for t in pytree.tree_leaves(call)
                   if isinstance(t, torch.Tensor) and t.is_cuda}, key=str)


def _capture(module: nn.Module, example: tuple, wanted: dict) -> dict:
    """{path: [(args, kwargs, output), ...]}: every call of the modules in ``wanted``
    ({module: path}) in one eager run of ``module`` on ``example`` (eval, no grad, the
    random state and every module's mode left as they were)."""
    calls = {p: [] for p in wanted.values()}
    hooks = [m.register_forward_hook(lambda m, a, k, o, p=p: calls[p].append((a, k, o)), with_kwargs=True)
             for m, p in wanted.items()]
    modes = [(m, m.training) for m in module.modules()]
    args, kwargs = example[0], example[1]
    try:
        module.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=_devices(example)):
            module(*args, **kwargs)
    finally:
        for h in hooks:
            h.remove()
        for m, mode in modes:
            m.training = mode
    return calls


class _Region:
    """One module call traced by one tracer, before inlining."""

    def __init__(self, kind, path, graph, inputs, outputs, spec):
        self.kind, self.path, self.graph = kind, path, graph
        self.inputs = inputs            # placeholder -> graph value of the parent (or a constant)
        self.outputs, self.spec = outputs, spec     # output leaves (region nodes or constants)
        self.subs = {}                  # region node -> fn(env, scope) -> value: a call to open
        self.targets = {}               # region node -> its get_attr / call_module target in the model
        self.drop = set()               # region nodes not copied
        self.input_dims = None          # export of the root: what the input guard checks
        self.guards = {}                # placeholder -> (dims, earlier placeholders): input checks


# ── the builder ────────────────────────────────────────────────────────────────

class _Call(nn.Module):
    """``model(*args, **kwargs)`` with the dynamic leaves as positional inputs, returning
    its output's tensors (or ``out[key]`` for the root)."""

    def __init__(self, model, spec, leaves, dynamic, key=_MISSING):
        super().__init__()
        self.model = model
        self.spec, self.leaves, self.dynamic, self.key = spec, leaves, dynamic, key

    def forward(self, *values):
        leaves = list(self.leaves)
        for p, v in zip(self.dynamic, values):
            leaves[p] = v
        args, kwargs = pytree.tree_unflatten(leaves, self.spec)
        out = self.model(*args, **kwargs)
        if self.key is not _MISSING:
            return out if self.key is None else out[self.key]
        return tuple(o for o in pytree.tree_leaves(out) if isinstance(o, torch.Tensor))


class Builder:
    def __init__(self, network: nn.Module, prefer: str, expand):
        self.network = network
        self.order = ["export", "fx"] if prefer == "export" else ["fx", "export"]
        self.expand = expand           # set of module paths, or "all"
        self.graph = fx.Graph()
        self.consts: dict = {}         # get_attr targets that aren't the model's (export's constants)
        self.calls: dict = {}          # module path -> calls inlined so far (scope call index)
        self.kinds: set = set()
        self.input_dims = None
        self.outer = _chain(sys.exception())     # errors being handled by our caller

    # ── entry point ───────────────────────────────────────────────────────────

    def build(self, example_input, key):
        """Trace the model on ``example_input``; the graph's output is ``out[key]``
        (``key`` None: the output itself)."""
        net = self.network
        with torch.no_grad(), torch.random.fork_rng(devices=_devices(example_input)):
            was = net.training
            net.eval()
            try:
                out = net(example_input)
            finally:
                net.train(was)
        placeholder = self.graph.placeholder("input")
        example = ((example_input,), {}, out)
        was = net.training
        net.train()                    # the graph is the training-mode forward
        try:
            value = self.open(net, "", ((placeholder,), {}), example, self.order, self.expand, [], key=key)
        finally:
            net.train(was)
        if isinstance(value, _Leaf):
            raise TraceError(f"neither torch.fx nor torch.export could trace "
                             f"{type(net).__name__}.forward:\n  " + "\n  ".join(value.errors))
        if isinstance(value, _Struct):
            target = out if key is None else out[key]
            flat = pytree.tree_leaves(out)
            value = value.flat[next(i for i, v in enumerate(flat) if v is target)]
        self.graph.output(value)
        return self.graph

    # ── opening one module call ───────────────────────────────────────────────

    def open(self, module, path, call, example, order, expand, scope, key=_MISSING, rules=None):
        """Trace ``module`` called with ``call`` (parent graph values; ``example``: the same
        call in an eager run, with its output) and inline it. Returns the output's value:
        a graph value, a constant, or a ``_Struct``."""
        errors = []
        for kind in order:
            try:
                if kind == "fx":
                    region = self._fx_region(module, path, call, example, expand)
                else:
                    region = self._export_region(module, path, call, example, key)
            except _Untraceable as e:
                errors.append(f"torch.{kind}: {e}")
                continue
            self.kinds.add(kind)
            if path == "" and region.input_dims is not None:
                self.input_dims = region.input_dims
            outs = self._inline(region, scope)
            return outs[0] if region.spec.is_leaf() else _Struct(region.spec, outs)
        if path == "":
            return _Leaf(errors)
        return self._leaf(path, call, example, scope, rules)

    def _leaf(self, path, call, example, scope, rules):
        args, kwargs = fx.node.map_aggregate(call, self._value)
        node = self.graph.call_module(path, tuple(args), dict(kwargs))
        node.meta.update(ts_kind="leaf", ts_scope=list(scope))
        if rules is not None:
            node.meta["ts_check"] = rules
        self.kinds.add("leaf")
        leaves, spec = pytree.tree_flatten(example[2])
        if spec.is_leaf():
            return node
        flat = []
        for i, leaf in enumerate(leaves):
            if isinstance(leaf, torch.Tensor):
                item = self.graph.call_function(_leaf_item, (node, i))
                item.meta.update(ts_kind="fx", ts_scope=list(scope))
                flat.append(item)
            else:
                flat.append(leaf)
        return _Struct(spec, flat)

    def _value(self, v):
        """A graph value for ``v`` (a ``_Struct`` is rebuilt by a node)."""
        if isinstance(v, _Struct):
            node = self.graph.call_function(_rebuild, (v.spec, *[self._value(x) for x in v.flat]))
            node.meta.update(ts_kind="fx", ts_scope=[])
            return node
        return v

    # ── fx ────────────────────────────────────────────────────────────────────

    def _fx_region(self, module, path, call, example, expand) -> _Region:
        args, kwargs = call
        sig = inspect.signature(module.forward)
        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError as e:
            raise _Untraceable(_short(e))
        concrete, dynamic = {}, {}
        for name, p in sig.parameters.items():
            variadic = p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            if name not in bound.arguments:
                if not variadic and p.default is not p.empty:
                    concrete[name] = p.default
                continue
            v = bound.arguments[name]
            if variadic:
                if v:
                    raise _Untraceable(f"{type(module).__name__}.forward is given *args/**kwargs")
                continue
            if isinstance(v, fx.Node):
                dynamic[name] = v
            elif any(isinstance(x, (fx.Node, _Struct)) for x in _leaves(v)):
                raise _Untraceable(f"argument {name!r} is a structure of tensors")
            else:
                concrete[name] = v

        subtree = {id(m): _join(path, n) for n, m in module.named_modules() if n}
        blocked = set()
        while True:
            tracer = _RegionTracer(path, expand, blocked)
            try:
                with _traced_factories():
                    graph = tracer.trace(module, concrete_args=concrete or None)
                break
            except Exception as e:
                culprit = _culprit(e, subtree, self.outer)
                if culprit is None or culprit in blocked:
                    raise _Untraceable(_short(e))
                blocked.add(culprit)     # its own code defeats fx: a leaf, opened with export

        nodes = list(graph.nodes)
        output = next(n for n in nodes if n.op == "output")
        outputs, spec = pytree.tree_flatten(output.args[0], is_leaf=lambda v: isinstance(v, fx.Node))
        live = {o for o in outputs if isinstance(o, fx.Node)}
        for n in reversed(nodes):
            if n in live:
                live.update(n.all_input_nodes)
        for n in nodes:
            target = n.target if isinstance(n.target, str) else getattr(n.target, "__name__", "")
            if n not in live and n.op in ("call_function", "call_method") \
                    and target.endswith("_") and not target.endswith("__"):
                raise _Untraceable(f"the in-place op {target!r} has an unused result")

        inputs = {}
        for n in nodes:
            if n.op != "placeholder":
                continue
            if n.target in dynamic:
                inputs[n] = dynamic[n.target]
            else:                        # specialized (concrete_args) or an unused *args/**kwargs
                inputs[n] = n.args[0] if n.args else ({} if n.target.startswith("**") else ())
        region = _Region("fx", path, graph, inputs, outputs, spec)
        region.stack = lambda n: [(k.partition("@")[0], int(k.partition("@")[2] or 0))
                                  for k in (n.meta.get("nn_module_stack") or {})]
        calls_of = {}
        for n in nodes:
            if n.op in ("call_module", "get_attr"):
                region.targets[n] = _join(path, n.target)
            if n.op == "call_module" and _join(path, n.target) in blocked:
                fqn = _join(path, n.target)
                k = calls_of[fqn] = calls_of.get(fqn, -1) + 1
                region.subs[n] = self._fx_sub(module, example, blocked, fqn, k, n)
        return region

    def _fx_sub(self, module, example, blocked, fqn, k, node):
        """Open the k-th call of ``fqn`` (fx couldn't) with export."""
        def open_sub(env, scope):
            sub = self.network.get_submodule(fqn)
            if not hasattr(self, "_caps") or self._caps[0] is not example:
                wanted = {self.network.get_submodule(p): p for p in blocked}
                self._caps = (example, _capture(module, example, wanted))
            calls = self._caps[1][fqn]
            if k >= len(calls):
                raise TraceError(f"{fqn} was called {len(calls)} times in an eager run, "
                                 f"but traced {k + 1} times")
            call = fx.node.map_arg((node.args, node.kwargs), lambda a: env[a])
            return self.open(sub, fqn, call, calls[k], ["export"], "all", scope)
        return open_sub

    # ── export ────────────────────────────────────────────────────────────────

    def _export_region(self, module, path, call, example, key) -> _Region:
        from torch.export import Dim
        from .export_graph import _SIZE_READS, _input_dims, _localize_sizes, _op_name

        values = _leaves(call)
        leaves, spec = pytree.tree_flatten((example[0], example[1]))
        if len(values) != len(leaves):
            raise _Untraceable("the traced call doesn't match the eager one")
        dynamic = [i for i, v in enumerate(values) if isinstance(v, fx.Node)]
        consts = [None if i in dynamic else v for i, v in enumerate(values)]
        wrapper = _Call(module, spec, consts, dynamic, key)
        inputs = tuple(leaves[i] for i in dynamic)
        shapes = (tuple({d: Dim.AUTO for d in range(v.dim())} if isinstance(v, torch.Tensor) else Dim.AUTO
                        for v in inputs),)
        subtree = {id(m): _join(path, n) for n, m in module.named_modules() if n}
        opaque, caps = set(), {}
        while True:
            saved = self._patch(opaque, caps, module, example)
            err = io.StringIO()
            try:
                with contextlib.redirect_stderr(err):   # a failed export prints its partial graph
                    ep = torch.export.export(wrapper, inputs, dynamic_shapes=shapes, strict=False)
                sys.stderr.write(err.getvalue())
                break
            except Exception as e:
                partial = str(getattr(e, "partial_fx_graph", "") or "")
                sys.stderr.write(err.getvalue().replace(partial + "\n", "") if partial else err.getvalue())
                culprit = _culprit(e, subtree, self.outer)
                if culprit is None or culprit in opaque:
                    raise _Untraceable(_short(e))
                opaque.add(culprit)      # its own code defeats export: opaque, opened with fx
            finally:
                for m in saved:
                    del m.forward

        gm = ep.module()
        graph = gm.graph
        _localize_sizes(graph)
        for n in list(graph.nodes):        # a size is read where its tensor is made: only the
            if n.op == "call_function" and _op_name(n.target) in _SIZE_READS \
                    and n.args[0].op != "get_attr":
                n.args[0].append(n)        # number crosses a cut, not the tensor
        nodes = list(graph.nodes)
        placeholders = [n for n in nodes if n.op == "placeholder"]
        output = next(n for n in nodes if n.op == "output")
        produced = list(output.args[0])
        if key is not _MISSING:
            outputs, out_spec = produced[:1], pytree.tree_structure(0)
        else:
            out_leaves, out_spec = pytree.tree_flatten(example[2])
            it = iter(produced)
            outputs = [next(it) if isinstance(v, torch.Tensor) else v for v in out_leaves]
        region = _Region("export", path, graph, {p: values[i] for p, i in zip(placeholders, dynamic)},
                         outputs, out_spec)
        if path == "":
            region.input_dims = _input_dims(placeholders[0], ep)
        else:      # the sizes this export assumed for its inputs, checked at run time
            seen = {}
            for p in placeholders:
                dims = _input_dims(p, ep, seen)
                refs = sorted({d[1] for d in dims if isinstance(d, tuple) and d[0] == "="},
                              key=placeholders.index)
                dims = [("=", refs.index(d[1]), d[2]) if isinstance(d, tuple) and d[0] == "=" else d
                        for d in dims]
                region.guards[p] = (dims, refs)

        def strip(p: str) -> str:
            return p[len("model."):] if p.startswith("model.") else ("" if p == "model" else p)

        def stack(n):
            out = []
            for k, (p, _) in (n.meta.get("nn_module_stack") or {}).items():
                rel = strip(p)
                if rel and not rel.startswith("_empty_nn_module_stack") and p != "model":
                    out.append((rel, int(k.rpartition("@")[2]) if "@" in k else 0))
            return out
        region.stack = stack

        for n in nodes:
            if n.op == "call_module":
                region.drop.add(n)                     # export's ``_guards_fn``
            elif n.op == "get_attr":
                fqn = _join(path, strip(n.target))
                value = gm
                for part in n.target.split("."):
                    value = getattr(value, part)
                if not self._owns(fqn, value):
                    while fqn in self.consts and self.consts[fqn] is not value:
                        fqn += "'"
                    self.consts[fqn] = value
                region.targets[n] = fqn
            elif n.op == "call_function" and n.target is torch.ops.torchslicer.opaque.default:
                region.subs[n] = self._export_sub(_OPAQUE[n.args[0]], n, caps)
        return region

    def _owns(self, fqn, value) -> bool:
        for get in (self.network.get_parameter, self.network.get_buffer):
            try:
                return get(fqn) is value
            except (AttributeError, KeyError, ValueError):
                pass
        return False

    def _patch(self, opaque, caps, module, example) -> list:
        """Make each module in ``opaque`` call the opaque op for the export to come."""
        if not opaque:
            return []
        missing = {self.network.get_submodule(p): p for p in opaque if p not in caps}
        if missing:
            caps.update(_capture(module, example, missing))
        patched = []
        for fqn in opaque:
            m = self.network.get_submodule(fqn)
            m.forward = self._opaque_forward(fqn, caps[fqn])
            patched.append(m)
        return patched

    @staticmethod
    def _opaque_forward(fqn, calls):
        count = [0]

        def forward(*args, **kwargs):
            k = count[0]
            count[0] += 1
            if k >= len(calls):
                raise TraceError(f"{fqn} was called {len(calls)} times in an eager run, "
                                 f"but traced {k + 1} times")
            e_args, e_kwargs, e_out = calls[k]
            leaves, spec = pytree.tree_flatten((args, kwargs))
            e_leaves = pytree.tree_leaves((e_args, e_kwargs))
            tensors = [i for i, v in enumerate(leaves) if isinstance(v, torch.Tensor)]
            ints = [i for i, v in enumerate(leaves) if isinstance(v, torch.SymInt)]
            out_leaves, out_spec = pytree.tree_flatten(e_out)
            outs = [o for o in out_leaves if isinstance(o, torch.Tensor)]
            device = outs[0].device if outs else torch.device("cpu")
            info = _Opaque(fqn, k, spec, [None if i in tensors or i in ints else v for i, v in enumerate(leaves)],
                           tensors, ints, out_spec, out_leaves, _shape_rules(e_leaves, tensors + ints, outs),
                           device)
            key = len(_OPAQUE)
            _OPAQUE[key] = info
            result = iter(torch.ops.torchslicer.opaque(key, [leaves[i] for i in tensors],
                                                       [leaves[i] for i in ints]))
            flat = [next(result) if isinstance(v, torch.Tensor) else v for v in out_leaves]
            return pytree.tree_unflatten(flat, out_spec)
        return forward

    def _export_sub(self, info, node, caps):
        """Open a module call export couldn't trace (the opaque op ``node``) with fx."""
        def open_sub(env, scope):
            leaves = info.fill([env[a] for a in node.args[1]], [env[a] for a in node.args[2]])
            args, kwargs = pytree.tree_unflatten(leaves, info.spec)
            sub = self.network.get_submodule(info.path)
            value = self.open(sub, info.path, (args, kwargs), caps[info.path][info.k], ["fx"], "all",
                              scope, rules=info.rules)
            if not isinstance(value, _Struct):
                return [value]
            return [v for v, o in zip(value.flat, info.out_leaves) if isinstance(o, torch.Tensor)]
        return open_sub

    # ── inlining ──────────────────────────────────────────────────────────────

    def _inline(self, region: _Region, scope: list) -> list:
        env = dict(region.inputs)
        offsets, seen = dict(self.calls), {}
        for p, (dims, refs) in region.guards.items():
            check = self.graph.call_function(_check_input, (env[p], dims, f"input {list(region.inputs).index(p)} of {region.path}",
                                                            *[env[r] for r in refs]))
            check.meta.update(ts_kind="fx", ts_scope=list(scope))
            env[p] = check

        def scope_of(n) -> list:
            out = list(scope)
            for rel, c in region.stack(n):
                fqn = _join(region.path, rel)
                seen[fqn] = max(seen.get(fqn, -1), c)
                g = c + offsets.get(fqn, 0)
                out.append(f"{fqn}:{g}" if g else fqn)
            return out

        for n in region.graph.nodes:
            if n.op in ("placeholder", "output") or n in region.drop:
                continue
            if n in region.subs:
                env[n] = region.subs[n](env, scope_of(n))
                continue
            if n.op == "call_function" and n.target in (operator.getitem, getattr):
                base = env.get(n.args[0])
                if isinstance(base, list):                       # an opaque op's outputs
                    env[n] = base[n.args[1]]
                    continue
                if isinstance(base, _Struct):
                    v = base.get(n.args[1])
                    if v is not _MISSING:
                        env[n] = v
                        continue
            new = self.graph.node_copy(n, lambda a: self._value(env[a]))
            new.meta["ts_kind"] = region.kind
            new.meta["ts_scope"] = scope_of(n)
            if n in region.targets:
                new.target = region.targets[n]
            env[n] = new
        for fqn, c in seen.items():
            self.calls[fqn] = max(self.calls.get(fqn, 0), offsets.get(fqn, 0) + c + 1)
        return [env[o] if isinstance(o, fx.Node) else o for o in region.outputs]


class _Leaf:
    """The root module, which no tracer could open."""

    def __init__(self, errors):
        self.errors = errors


class _RegionTracer(fx.Tracer):
    """fx with the leaves ``expand`` asks for (paths from the model's root), and the
    modules in ``blocked`` (fx failed on their code) as leaves too."""

    def __init__(self, path, expand, blocked):
        super().__init__()
        self.path, self.expand, self.blocked = path, expand, blocked

    def is_leaf_module(self, m: nn.Module, qualified_name: str) -> bool:
        fqn = _join(self.path, qualified_name)
        if fqn in self.blocked:
            return True
        if self.expand == "all":
            return super().is_leaf_module(m, qualified_name)
        return fqn not in self.expand
