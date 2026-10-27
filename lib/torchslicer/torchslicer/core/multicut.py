"""Multi-tensor cuts: several values cross one cut, carried as one tensor.

Every executor, transport, hook and metric moves exactly one tensor across each cut. A cut
that several values must cross (a skip connection, a U-Net level, the model input read late)
is rewritten in the graph instead: a ``send`` node at the end of the slice packs every value
that crosses into one tensor, and one ``recv`` node per value at the start of the next slice
takes it apart. Both are differentiable, so the gradient of the packed tensor is the packed
gradients, and the rest of TorchSlicer sees an ordinary single-tensor cut.

Packed layout, ``[B, L + n_1 + ... + n_m]`` in one float dtype ``W``:

- ``B`` is the batch size (the first dimension of the first value); every row starts with the
  same ``L``-value header (bytes 0..255, exact in any float dtype) describing the values;
- a value whose first dimension is ``B`` is laid out per row (``reshape(B, -1)``), so rows are
  samples; any other value (a parameter, a ``[1, T, d]`` table, a Python number) is repeated
  in every row and read back from row 0 (autograd sums the rows' gradients);
- a ``torch.device`` crossing a cut is read back as the device of the receiving slice (where
  the value is used); a ``torch.dtype`` as itself;
- floating-point tensors are cast to ``W``, the widest float dtype among them (lossless, cast
  back on receipt); integer and bool tensors and Python numbers travel as their bytes, one
  byte per element of ``W``; ``None`` and tuples / lists of values (e.g. a module returning
  ``(output, weights)``) are described in the header.
"""
import copy
import struct

import torch
import torch.nn as nn

from .model_graph import (LayerNode, ModelGraph, _AddWrapper, _AttrWrapper, _FlattenWrapper,
                          _FunctionWrapper, _ReshapeWrapper)

_MAGIC, _VERSION = 0x5A, 1
_HEAD = struct.Struct("<BBH")             # magic, version, number of values
_TENSOR = struct.Struct("<BB")            # dtype code, ndim (after the kind byte)
_COUNT = struct.Struct("<H")              # items of a tuple / list
_DIM = struct.Struct("<I")
_ROWS, _REPEATED, _SCALAR, _NONE, _TUPLE, _LIST, _DEVICE, _DTYPE = range(8)     # kinds
_DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int64, torch.int32,
           torch.int16, torch.int8, torch.uint8, torch.bool]
_CODE = {d: i for i, d in enumerate(_DTYPES)}
_SCALARS = [bool, int, float]             # a Python number travels as the bytes of a 0-dim tensor
_SCALAR_DTYPE = {bool: torch.bool, int: torch.int64, float: torch.float64}
_PEEK = 1024                              # header values read before knowing its length


def _tensors(v):
    """The tensors in ``v`` (nested tuples / lists), in order."""
    if isinstance(v, torch.Tensor):
        return [v]
    if isinstance(v, (tuple, list)):
        return [t for x in v for t in _tensors(x)]
    return []


class _SendCut(nn.Module):
    """Packs the values crossing a cut into one tensor (the slice's output). A value is a
    tensor, a Python number, ``None``, or a tuple / list of those."""

    def forward(self, *values):
        tensors = [t for v in values for t in _tensors(v)]
        first = tensors[0] if tensors else torch.zeros(())
        batch = first.shape[0] if first.dim() > 0 else 1
        floats = [t.dtype for t in tensors if t.is_floating_point()]
        wire = floats[0] if floats else torch.float32
        for d in floats[1:]:
            wire = torch.promote_types(wire, d)
        device = next((t.device for t in tensors if t.device.type != "cpu"), first.device)
        head = bytearray(_HEAD.pack(_MAGIC, _VERSION, len(values)))
        cols = []

        def leaf(t, kind, as_bytes):
            flat = t.to(device).reshape(batch, -1) if kind == _ROWS else t.to(device).reshape(1, -1)
            flat = flat.contiguous().view(torch.uint8).to(wire) if as_bytes else flat.to(wire)
            cols.append(flat if kind == _ROWS else flat.expand(batch, -1))

        def put(v):
            nonlocal head
            if v is None:
                head.append(_NONE)
            elif isinstance(v, torch.device):         # read back as the receiving side's device
                head.append(_DEVICE)
            elif isinstance(v, torch.dtype):
                if v not in _CODE:
                    raise TypeError(f"unsupported dtype value on a multi-tensor cut: {v}")
                head += bytes([_DTYPE, _CODE[v]])
            elif isinstance(v, (tuple, list)):
                head.append(_TUPLE if isinstance(v, tuple) else _LIST)
                head += _COUNT.pack(len(v))
                for x in v:
                    put(x)
            elif type(v) in _SCALAR_DTYPE:
                head += bytes([_SCALAR, _SCALARS.index(type(v))])
                leaf(torch.tensor(v, dtype=_SCALAR_DTYPE[type(v)]), _SCALAR, True)
            elif isinstance(v, torch.Tensor):
                if v.dtype not in _CODE:
                    raise TypeError(f"unsupported dtype on a multi-tensor cut: {v.dtype}")
                rows = v.dim() > 0 and v.shape[0] == batch
                shape = list(v.shape[1:] if rows else v.shape)
                kind = _ROWS if rows else _REPEATED
                head.append(kind)
                head += _TENSOR.pack(_CODE[v.dtype], len(shape)) + b"".join(_DIM.pack(d) for d in shape)
                leaf(v, kind, not v.is_floating_point())
            else:
                raise TypeError("a multi-tensor cut carries tensors, Python numbers, None and "
                                f"tuples / lists of them, not {type(v).__name__}")

        for v in values:
            put(v)
        header = torch.frombuffer(head, dtype=torch.uint8).to(device=device, dtype=wire)
        return torch.cat([header.expand(batch, -1), *cols], dim=1)


class _Reader:
    """Reads the header bytes of row 0 of a packed tensor, fetching more when needed."""

    def __init__(self, packed):
        self.packed, self.off = packed, 0
        self.raw = packed[0, :_PEEK].detach().to(torch.uint8).cpu().numpy().tobytes()

    def take(self, fmt: struct.Struct):
        if self.off + fmt.size > len(self.raw):
            self.raw = self.packed[0].detach().to(torch.uint8).cpu().numpy().tobytes()
        out = fmt.unpack_from(self.raw, self.off)
        self.off += fmt.size
        return out


_BYTE = struct.Struct("<B")


def _layout(packed):
    """The values of ``packed`` as specs (nested), read from row 0 once and cached on the
    tensor (every ``recv`` node of a slice reads the same object). A leaf spec is
    ``(kind, dtype, shape, start, width)``; a tuple / list spec is ``(kind, [specs])``."""
    cached = getattr(packed, "_ts_layout", None)
    if cached is not None:
        return cached
    r = _Reader(packed)
    magic, version, n = r.take(_HEAD)
    if magic != _MAGIC or version != _VERSION:
        raise ValueError("the tensor received at this cut is not a packed multi-tensor cut "
                         "(a cut hook changed it?)")
    leaves = []

    def get():
        kind, = r.take(_BYTE)
        if kind in (_NONE, _DEVICE):
            return (kind,)
        if kind == _DTYPE:
            return (kind, _DTYPES[r.take(_BYTE)[0]])
        if kind in (_TUPLE, _LIST):
            count, = r.take(_COUNT)
            return (kind, [get() for _ in range(count)])
        if kind == _SCALAR:
            code, = r.take(_BYTE)
            spec = [kind, _SCALAR_DTYPE[_SCALARS[code]], [], 0, 0]
        else:
            code, ndim = r.take(_TENSOR)
            spec = [kind, _DTYPES[code], [r.take(_DIM)[0] for _ in range(ndim)], 0, 0]
        leaves.append(spec)
        return spec

    values = [get() for _ in range(n)]
    start = r.off
    for spec in leaves:
        _, dtype, shape, _, _ = spec
        numel = 1
        for d in shape:
            numel *= d
        width = numel if dtype.is_floating_point and spec[0] != _SCALAR \
            else numel * torch.empty(0, dtype=dtype).element_size()
        spec[3], spec[4] = start, width
        start += width
    packed._ts_layout = values
    return values


def _build(packed, spec):
    kind = spec[0]
    if kind == _NONE:
        return None
    if kind == _DEVICE:
        return packed.device
    if kind == _DTYPE:
        return spec[1]
    if kind in (_TUPLE, _LIST):
        items = [_build(packed, s) for s in spec[1]]
        return tuple(items) if kind == _TUPLE else items
    _, dtype, shape, start, width = spec
    cols = packed[:, start:start + width] if kind == _ROWS else packed[:1, start:start + width]
    if dtype.is_floating_point and kind != _SCALAR:
        t = cols.to(dtype)
    else:                                  # bytes: an own contiguous buffer to view as dtype
        t = cols.detach().to(torch.uint8).flatten().clone().view(dtype)
    if kind == _ROWS:
        return t.reshape(packed.shape[0], *shape)
    t = t.reshape(shape)
    return t.item() if kind == _SCALAR else t


class _RecvCut(nn.Module):
    """Value ``index`` of a packed cut."""

    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, packed):
        return _build(packed, _layout(packed)[self.index])

    def extra_repr(self):
        return str(self.index)


# ── the graph rewrite ──────────────────────────────────────────────────────────

_META_METHODS = {"size", "dim", "ndimension", "numel", "nelement", "get_device"}
_META_ATTRS = {"shape", "device", "dtype", "ndim", "is_cuda"}
_RANDOM = ("rand", "dropout", "bernoulli", "multinomial", "normal", "poisson", "uniform", "exponential")


def _is_meta(module) -> bool:
    """Reads only a tensor's metadata (``x.size(1)``, ``x.device``): no values, no gradient."""
    if not isinstance(module, _FunctionWrapper):
        return False
    if module.kind == "call_method":
        return module.target in _META_METHODS
    return (module.target is getattr and len(module.args) == 2
            and module.args[1] in _META_ATTRS)


def _is_pure(module) -> bool:
    """A traced op that computes the same thing wherever it runs, from its inputs only: a
    library wrapper (not a user module, which may keep state), not random, no parameter."""
    if isinstance(module, _AttrWrapper):
        return not isinstance(module.value, nn.Parameter)
    if isinstance(module, (_FlattenWrapper, _ReshapeWrapper, _AddWrapper, nn.Identity)):
        return True
    if isinstance(module, _FunctionWrapper):
        name = module.target if isinstance(module.target, str) else getattr(module.target, "__name__", "")
        return not any(r in name.lower() for r in _RANDOM)
    return False


def recomputable(graph) -> dict:
    """``{index: "meta" | "free"}``: metadata reads, and values computed from metadata and
    constants only (positions, causal masks, RoPE tables built in ``forward``). A free value
    that crosses a cut is rebuilt in the receiving slice; only the metadata it needs crosses."""
    kind = {}
    for node in graph.nodes:
        if _is_meta(node.module):
            kind[node.index] = "meta"
        elif (_is_pure(node.module) and not node.takes_input
              and all(p in kind for p in node.predecessors)):
            kind[node.index] = "free"
    return kind


def _split_free(values, nodes, kind):
    """What crosses instead of ``values``: free values are replaced by the metadata and
    non-free values they are built from; returns (sent, free nodes to rebuild)."""
    sent, rebuild, stack = [], set(), list(reversed(values))
    while stack:
        v = stack.pop()
        if v >= 0 and kind.get(v) == "free":
            if v not in rebuild:
                rebuild.add(v)
                stack.extend(reversed(nodes[v].predecessors))
        elif v not in sent:
            sent.append(v)
    if not sent:                       # something must cross: the main value, as it is
        return [values[0]], []
    return sent, sorted(rebuild)


def crossing(graph, k: int) -> list:
    """Nodes whose output a node after ``k`` reads (``k`` itself first when it is one of
    them); ``-1`` = the model input, read directly by a node after ``k``."""
    found = set()
    for node in graph.nodes[k + 1:]:
        found.update(p for p in node.predecessors if p <= k)
        if node.takes_input:
            found.add(-1)
    return sorted(found, key=lambda p: (p != k, p))


def is_packed_cut(graph, k: int) -> bool:
    """True if a cut after node ``k`` of a rewritten graph is a ``send`` node."""
    return isinstance(graph.nodes[k].module, _SendCut)


def packed_cuts(graph, partitions) -> set:
    """Cut indices ``k`` (slice ``k`` -> ``k+1``) that carry a packed multi-tensor value."""
    return {k for k, p in enumerate(partitions[:-1]) if is_packed_cut(graph, p.layer_indices[-1])}


def refuse_at_packed_cuts(graph, partitions, cuts, what: str) -> None:
    """Cut hooks and privacy observers see one tensor per cut; at a packed cut that tensor is
    the packed layout (header + every value), so they are refused there for now."""
    bad = sorted(set(cuts) & packed_cuts(graph, partitions))
    if bad:
        raise ValueError(f"{what} at cut {bad[0]}: several tensors cross it (multi_tensor=True), "
                         f"packed into one; {what} on packed cuts aren't supported yet. Use a cut "
                         f"one tensor crosses (ts.trace(model).valid_cuts()) for this one.")


def nearest_valid_cuts(graph, targets: list) -> list:
    """For each target gap (increasing), the nearest of ``graph.valid_cuts()``, kept distinct
    and increasing; ``ValueError`` when there are fewer valid cuts than targets."""
    valid = graph.valid_cuts()
    if len(valid) < len(targets):
        raise ValueError(f"{len(targets) + 1} slices need {len(targets)} cuts; this graph has "
                         f"{len(valid)} valid ones ({valid})")
    out, lo = [], 0
    for i, t in enumerate(targets):
        hi = len(valid) - (len(targets) - i)          # leave room for the remaining targets
        j = min(range(lo, hi + 1), key=lambda j: (abs(valid[j] - t), j))
        out.append(valid[j])
        lo = j + 1
    return out


def pack_cuts(graph, cuts: list, recompute: bool = True):
    """``(graph', cuts')``: ``graph`` with a send node after each cut that more than one value
    crosses (or whose single value is not the node right before it) and recv nodes after it;
    ``cuts'`` are the same cuts in ``graph'`` (after each send node). Nodes keep their modules,
    so the new graph shares every parameter with ``graph``. With ``recompute``, a crossing
    value built only from metadata and constants (see :func:`recomputable`) is rebuilt by a
    copy of its nodes after the cut, and only the metadata crosses. Cuts that are already
    valid are left alone, so a graph whose cuts are all valid comes back unchanged."""
    cuts = sorted(cuts)
    kind = recomputable(graph) if recompute else {}
    plan, rebuild = {}, {}
    for c in cuts:
        plan[c], rebuild[c] = _split_free(crossing(graph, c), graph.nodes, kind)
    if all(plan[c] == [c] and not rebuild[c] for c in cuts):
        return graph, cuts
    nodes = graph.nodes
    late_input = any(n.takes_input for n in nodes[cuts[0] + 1:]) if cuts else False
    new = ModelGraph(source=graph.source)
    new.expand, new.model = graph.expand, graph.model
    current: dict = {}                      # old value id -> new node index providing it here

    def add(module, name, preds, takes_input=False, scope=()):
        idx = len(new.nodes)
        new.nodes.append(LayerNode(index=idx, module=module, predecessors=list(preds),
                                   name=name, takes_input=takes_input, scope=list(scope)))
        return idx

    if late_input:                          # the model input crosses a cut: make it a node
        current[-1] = add(nn.Identity(), _unique(graph, "input"), [], takes_input=True)
    new_cuts, bounds = [], [-1, *cuts, len(nodes) - 1]
    for j in range(len(bounds) - 1):
        for i in range(bounds[j] + 1, bounds[j + 1] + 1):
            node = nodes[i]
            preds = [current[p] for p in node.predecessors]
            reads_input = node.takes_input and not late_input
            if node.takes_input and late_input:
                preds = [current[-1]] + preds if not preds else preds
            current[i] = add(node.module, node.name, preds, reads_input, node.scope)
        if j == len(bounds) - 2:
            break
        c = bounds[j + 1]
        values = plan[c]
        if values == [c]:
            new_cuts.append(current[c])
            here = {c: current[c]}                  # the slice's input, read as it is
        else:
            send = add(_SendCut(), _unique(graph, f"cut{j}.send"), [current[v] for v in values])
            new_cuts.append(send)
            here = {v: add(_RecvCut(vi), _unique(graph, f"cut{j}.recv{vi}"), [send])
                    for vi, v in enumerate(values)}
        for v in rebuild[c]:                        # free values: rebuilt after the cut
            node = nodes[v]
            here[v] = add(copy.deepcopy(node.module), _unique(graph, f"{node.name}.cut{j}"),
                          [here[p] for p in node.predecessors], scope=node.scope)
        current.update(here)
    for node in new.nodes:
        for p in node.predecessors:
            new.nodes[p].successors.append(node.index)
    new.base_cuts = cuts                    # recorded in the run manifest to rebuild this graph
    return new, new_cuts


def _unique(graph, name: str) -> str:
    names = set(graph.names())
    while name in names:
        name = "_" + name
    return name
