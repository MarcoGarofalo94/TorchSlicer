"""Cut hooks: change or observe what crosses a cut.

Cut ``k`` is the boundary between slice ``k`` and slice ``k + 1``. Forward, slice ``k``
sends its output (an activation) to slice ``k + 1``; backward, slice ``k + 1`` sends the
gradient of that activation back. A hook sits on one cut and has four methods, each
called on one side of it:

==============  ==========================  =========================================
method          runs on                     does
==============  ==========================  =========================================
``encode``      slice ``k`` (sender)        activation -> payload put on the wire
``decode``      slice ``k + 1`` (receiver)  payload -> activation slice ``k + 1`` uses
``encode_grad`` slice ``k + 1`` (sender)    gradient -> payload
``decode_grad`` slice ``k`` (receiver)      payload -> gradient slice ``k`` backprops
==============  ==========================  =========================================

A payload is a tensor or a list of tensors. Every method defaults to the identity.
The gradient slice ``k + 1`` computes is with respect to what it received, the
decoded activation, and slice ``k`` backpropagates the decoded gradient through its
own output (straight-through): this is what a lossy cut means in split learning.

In a distributed run the two sides are different processes, each with its own copy of
the hook (shipped with the slice, so a custom hook class must be importable on the
workers, like a custom layer). Hook state (an error-feedback residual, a recorder's
records) is per side. In ``LocalExecutor`` both sides use the object you passed.

Built in: :class:`Float16`, :class:`BFloat16`, :class:`Quantize`, :class:`TopK`,
:class:`GaussianNoise`, :class:`Recorder`, :class:`Chain`. Config files name them
(``sl.cut_hooks``, see :func:`hooks_from_config`); ``register_hook`` adds your own.
"""

import os
import struct
from dataclasses import dataclass

import torch


@dataclass
class CutContext:
    cut:       int        # boundary index: between slice ``cut`` and ``cut + 1``
    direction: str        # "forward" (activation) | "backward" (gradient)
    batch_id:  int        # micro-batch id under GPipe
    n_slices:  int = 0


class CutHook:
    """Base class: identity on both directions. Override what you need."""

    def encode(self, x: torch.Tensor, ctx: CutContext):
        return x

    def decode(self, payload, ctx: CutContext) -> torch.Tensor:
        return payload

    def encode_grad(self, g: torch.Tensor, ctx: CutContext):
        return g

    def decode_grad(self, payload, ctx: CutContext) -> torch.Tensor:
        return payload


# ── payload on the wire ─────────────────────────────────────────────────────────

_DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.int64,
           torch.int32, torch.uint8, torch.int8, torch.bool, torch.int16]
_CODE = {d: i for i, d in enumerate(_DTYPES)}
_HEAD = struct.Struct("!BB")       # is_list, n_tensors
_TENSOR_HEAD = struct.Struct("!BB")  # dtype code, ndim
_DIM = struct.Struct("!q")


def _as_list(payload) -> tuple[bool, list]:
    if isinstance(payload, torch.Tensor):
        return False, [payload]
    if isinstance(payload, (list, tuple)) and all(isinstance(t, torch.Tensor) for t in payload):
        if len(payload) > 255:
            raise ValueError("a cut payload holds at most 255 tensors")
        return True, list(payload)
    raise TypeError(f"a cut payload is a tensor or a list of tensors, got {type(payload).__name__}")


def payload_nbytes(payload) -> int:
    """Bytes the payload takes on the wire once packed (``pack_payload``)."""
    _, ts = _as_list(payload)
    return _HEAD.size + sum(_TENSOR_HEAD.size + _DIM.size * t.dim() + t.numel() * t.element_size()
                            for t in ts)


def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def pack_payload(payload) -> torch.Tensor:
    """One uint8 tensor holding the payload (header + raw buffers), on the payload's device:
    a CUDA payload stays on the GPU, so a GPU transport (``ipc``, ``nccl``) can carry it."""
    is_list, ts = _as_list(payload)
    device = next((t.device for t in ts if t.numel()), torch.device("cpu"))
    head = bytearray(_HEAD.pack(int(is_list), len(ts)))
    bufs = []
    for t in ts:
        if t.dtype not in _CODE:
            raise TypeError(f"unsupported dtype on a cut: {t.dtype}")
        t = t.detach().to(device).contiguous()
        head += _TENSOR_HEAD.pack(_CODE[t.dtype], t.dim())
        head += b"".join(_DIM.pack(int(d)) for d in t.shape)
        bufs.append(t.reshape(-1).view(torch.uint8) if t.numel()
                    else torch.empty(0, dtype=torch.uint8, device=device))
    return torch.cat([torch.frombuffer(head, dtype=torch.uint8).to(device), *bufs])


def _parse_head(raw: bytes):
    """(is_list, [(dtype, shape)], header size) from the start of a packed payload;
    ``struct.error`` when ``raw`` is shorter than the header."""
    is_list, n = _HEAD.unpack_from(raw, 0)
    off = _HEAD.size
    shapes = []
    for _ in range(n):
        code, ndim = _TENSOR_HEAD.unpack_from(raw, off)
        off += _TENSOR_HEAD.size
        shape = [_DIM.unpack_from(raw, off + i * _DIM.size)[0] for i in range(ndim)]
        off += ndim * _DIM.size
        shapes.append((_DTYPES[code], shape))
    return is_list, shapes, off


_HEAD_PEEK = 4096   # bytes copied to the host to parse the header of a device payload


def unpack_payload(packed: torch.Tensor, device=None):
    """Inverse of ``pack_payload`` (tensors land on ``device``, default the packed one's).
    Only the header is copied to the host; the buffers are sliced where ``packed`` lives."""
    data = packed.detach().contiguous()
    try:
        is_list, shapes, off = _parse_head(data[:_HEAD_PEEK].cpu().numpy().tobytes())
    except struct.error:   # a header longer than the peek (many tensors of high rank)
        is_list, shapes, off = _parse_head(data.cpu().numpy().tobytes())
    out = []
    for dtype, shape in shapes:
        numel = 1
        for d in shape:
            numel *= d
        nb = numel * torch.empty(0, dtype=dtype).element_size()
        # clone: an aligned buffer of its own (the offset may not suit ``dtype``)
        t = data[off:off + nb].clone().view(dtype).reshape(shape)
        off += nb
        out.append(t.to(device) if device is not None else t)
    return out if is_list else out[0]


# ── built-in hooks ──────────────────────────────────────────────────────────────

class _Cast(CutHook):
    dtype = None

    def __init__(self, grad: bool = True):
        self.grad = grad

    def _down(self, t):
        return t.to(self.dtype) if t.is_floating_point() else t

    def encode(self, x, ctx):
        return self._down(x)

    def decode(self, payload, ctx):
        return payload.float() if payload.is_floating_point() else payload

    def encode_grad(self, g, ctx):
        return self._down(g) if self.grad else g

    def decode_grad(self, payload, ctx):
        return payload.float() if payload.is_floating_point() else payload


class Float16(_Cast):
    """Send activations (and, with ``grad``, gradients) as float16; the receiver
    casts back to float32. Halves the bytes of a float32 cut."""
    dtype = torch.float16


class BFloat16(_Cast):
    """Like :class:`Float16` with bfloat16 (wider range, less precision)."""
    dtype = torch.bfloat16


def _pack_bits(q: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 8:
        return q
    per = 8 // bits
    flat = q.reshape(-1)
    pad = (-flat.numel()) % per
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    flat = flat.reshape(-1, per).to(torch.int32)
    out = torch.zeros(flat.shape[0], dtype=torch.int32, device=q.device)
    for i in range(per):
        out |= flat[:, i] << (bits * i)
    return out.to(torch.uint8)


def _unpack_bits(packed: torch.Tensor, bits: int, numel: int) -> torch.Tensor:
    if bits == 8:
        return packed.reshape(-1)[:numel]
    per = 8 // bits
    mask = (1 << bits) - 1
    p = packed.to(torch.int32)
    vals = torch.stack([(p >> (bits * i)) & mask for i in range(per)], dim=1).reshape(-1)
    return vals[:numel].to(torch.uint8)


class Quantize(CutHook):
    """Uniform per-tensor quantization to ``bits`` (1, 2, 4 or 8) bits.

    ``x`` is mapped to ``round((x - min) / scale)`` with ``scale = (max - min) /
    (2**bits - 1)``; values are bit-packed (8/bits per byte) and sent with
    ``[min, scale]`` and the shape. ``grad`` quantizes the gradients too.
    """

    def __init__(self, bits: int = 8, grad: bool = True):
        if bits not in (1, 2, 4, 8):
            raise ValueError("Quantize(bits=) must be 1, 2, 4 or 8")
        self.bits = bits
        self.grad = grad

    def _enc(self, t):
        if not t.is_floating_point():
            return t
        tf = t.detach().float()
        lo, hi = tf.min(), tf.max()
        levels = (1 << self.bits) - 1
        scale = (hi - lo) / levels
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = torch.clamp(torch.round((tf - lo) / scale), 0, levels).to(torch.uint8)
        meta = torch.stack([lo, scale]).float()
        shape = torch.tensor(list(t.shape), dtype=torch.int64, device=t.device)
        return [_pack_bits(q, self.bits), meta, shape]

    def _dec(self, payload):
        if isinstance(payload, torch.Tensor):
            return payload
        packed, meta, shape = payload
        shape = [int(d) for d in shape.tolist()]
        numel = 1
        for d in shape:
            numel *= d
        q = _unpack_bits(packed, self.bits, numel).float()
        return (q * meta[1] + meta[0]).reshape(shape)

    def encode(self, x, ctx):
        return self._enc(x)

    def decode(self, payload, ctx):
        return self._dec(payload)

    def encode_grad(self, g, ctx):
        return self._enc(g) if self.grad else g

    def decode_grad(self, payload, ctx):
        return self._dec(payload)


class TopK(CutHook):
    """Keep the ``ratio`` largest-magnitude entries of each sample, zero the rest.

    Sends values, their int32 indices and the shape. ``grad`` sparsifies gradients
    too (off by default: the activation is what is usually compressed).
    """

    def __init__(self, ratio: float = 0.1, grad: bool = False):
        if not 0 < ratio <= 1:
            raise ValueError("TopK(ratio=) must be in (0, 1]")
        self.ratio = ratio
        self.grad = grad

    def _enc(self, t):
        if not t.is_floating_point() or t.dim() == 0:
            return t
        flat = t.detach().reshape(t.shape[0], -1)
        k = max(1, int(round(self.ratio * flat.shape[1])))
        idx = flat.abs().topk(k, dim=1).indices
        vals = flat.gather(1, idx)
        shape = torch.tensor(list(t.shape), dtype=torch.int64, device=t.device)
        return [vals, idx.to(torch.int32), shape]

    def _dec(self, payload):
        if isinstance(payload, torch.Tensor):
            return payload
        vals, idx, shape = payload
        shape = [int(d) for d in shape.tolist()]
        n = 1
        for d in shape[1:]:
            n *= d
        out = torch.zeros(shape[0], n, dtype=vals.dtype, device=vals.device)
        out.scatter_(1, idx.long(), vals)
        return out.reshape(shape)

    def encode(self, x, ctx):
        return self._enc(x)

    def decode(self, payload, ctx):
        return self._dec(payload)

    def encode_grad(self, g, ctx):
        return self._enc(g) if self.grad else g

    def decode_grad(self, payload, ctx):
        return self._dec(payload)


class GaussianNoise(CutHook):
    """Add N(0, std^2) noise to the activation before it leaves the sender (and to the
    gradient with ``grad``). The noise is drawn from a generator seeded with
    ``(seed, cut, batch_id, direction)``: it neither consumes nor depends on the global
    RNG (data order, dropout), and it is the same in every topology."""

    def __init__(self, std: float, grad: bool = False, seed: int = 0):
        self.std = float(std)
        self.grad = grad
        self.seed = seed

    def _noise(self, t, ctx):
        if not t.is_floating_point() or self.std == 0:
            return t
        # seeded per (seed, cut, batch, direction): the same noise in every topology
        gen = torch.Generator(device=t.device)
        gen.manual_seed((self.seed * 1_000_003 + ctx.cut * 7_919 + ctx.batch_id * 2
                         + (ctx.direction == "backward")) % (2 ** 63))
        return t + self.std * torch.randn(t.shape, generator=gen, device=t.device, dtype=t.dtype)

    def encode(self, x, ctx):
        return self._noise(x, ctx)

    def encode_grad(self, g, ctx):
        return self._noise(g, ctx) if self.grad else g


class Recorder(CutHook):
    """Observe a cut without changing it, e.g. as the input of a reconstruction or
    label-inference attack.

    Records CPU copies of what the *receiving* side gets (``what``: "forward" =
    activations, "backward" = gradients), for the first ``max_batches`` batches, in
    ``self.records`` as ``(direction, batch_id, tensor)``. With ``path``, each record
    is also saved to ``{path}/cut{k}_{direction}_{batch_id}.pt``: in a distributed run
    the hook runs on the worker, so files are how you get the records back.
    """

    def __init__(self, what=("forward", "backward"), max_batches: int = 10, path: str | None = None):
        self.what = (what,) if isinstance(what, str) else tuple(what)
        self.max_batches = max_batches
        self.path = path
        self.records = []

    def _record(self, t, ctx):
        if ctx.direction not in self.what:
            return
        if sum(1 for r in self.records if r[0] == ctx.direction) >= self.max_batches:
            return
        rec = t.detach().cpu().clone()
        self.records.append((ctx.direction, ctx.batch_id, rec))
        if self.path:
            os.makedirs(self.path, exist_ok=True)
            torch.save(rec, os.path.join(self.path, f"cut{ctx.cut}_{ctx.direction}_{ctx.batch_id}.pt"))

    def decode(self, payload, ctx):
        self._record(payload, ctx)
        return payload

    def decode_grad(self, payload, ctx):
        self._record(payload, ctx)
        return payload


class Chain(CutHook):
    """Several hooks on one cut: encoders run in order, decoders in reverse. Every
    hook but the last must keep the payload a single tensor (e.g. noise, then
    quantize)."""

    def __init__(self, hooks):
        self.hooks = list(hooks)

    @staticmethod
    def _check(p, h):
        if not isinstance(p, torch.Tensor):
            raise TypeError(f"Chain: {type(h).__name__} must produce a tensor when it is not last")
        return p

    def encode(self, x, ctx):
        for i, h in enumerate(self.hooks):
            x = h.encode(x, ctx)
            if i < len(self.hooks) - 1:
                self._check(x, h)
        return x

    def decode(self, payload, ctx):
        for h in reversed(self.hooks):
            payload = h.decode(payload, ctx)
        return payload

    def encode_grad(self, g, ctx):
        for i, h in enumerate(self.hooks):
            g = h.encode_grad(g, ctx)
            if i < len(self.hooks) - 1:
                self._check(g, h)
        return g

    def decode_grad(self, payload, ctx):
        for h in reversed(self.hooks):
            payload = h.decode_grad(payload, ctx)
        return payload


# ── resolution: Python / config spec → {cut: hook} ──────────────────────────────

_REGISTRY = {
    "float16": Float16,
    "bfloat16": BFloat16,
    "quantize": Quantize,
    "topk": TopK,
    "gaussian_noise": GaussianNoise,
    "recorder": Recorder,
}


def register_hook(name: str, cls) -> None:
    """Make ``cls`` available to config files as ``sl.cut_hooks: [{name: ...}]``."""
    _REGISTRY[name] = cls


def _one(h):
    if isinstance(h, CutHook):
        return h
    if isinstance(h, (list, tuple)):
        return Chain([_one(x) for x in h])
    raise TypeError(f"expected a CutHook (or a list of them), got {type(h).__name__}")


def hooks_from_config(entries) -> dict:
    """``[{cut: 0 | "all", name: "quantize", params: {bits: 8}}, ...]`` → ``{cut: hook}``
    (several entries on one cut are chained in order)."""
    out = {}
    for e in entries or []:
        name = e.get("name")
        if name not in _REGISTRY:
            raise ValueError(f"unknown cut hook {name!r}; known: {sorted(_REGISTRY)}")
        hook = _REGISTRY[name](**(e.get("params") or {}))
        cut = e.get("cut", "all")
        out.setdefault(cut, []).append(hook)
    return {c: (hs[0] if len(hs) == 1 else Chain(hs)) for c, hs in out.items()}


def resolve_cut_hooks(spec, n_slices: int, config_entries=None, drop_missing: bool = False) -> dict:
    """``{cut: CutHook}`` for a run with ``n_slices`` slices (cuts ``0 .. n_slices-2``).

    ``spec`` (from ``ts.slice(cut_hooks=)``) is a hook (every cut), a list of hooks
    (chained, every cut), or a dict ``{cut | "all": hook | [hooks]}``. Config entries
    (``sl.cut_hooks``) apply first; Python ones replace them cut by cut.
    ``drop_missing`` skips hooks on cuts that don't exist (recovery: fewer slices).
    """
    n_cuts = max(0, n_slices - 1)
    out = {}

    def _apply(mapping):
        for cut, h in mapping.items():
            targets = range(n_cuts) if cut == "all" else [int(cut)]
            for c in targets:
                if not 0 <= c < n_cuts and drop_missing:
                    continue
                if not 0 <= c < n_cuts:
                    raise ValueError(f"cut hook on cut {c}, but {n_slices} slices have cuts 0..{n_cuts - 1}")
                out[c] = _one(h)

    _apply(hooks_from_config(config_entries))
    if spec is None:
        pass
    elif isinstance(spec, dict):
        _apply(spec)
    else:
        _apply({"all": spec})
    return out
