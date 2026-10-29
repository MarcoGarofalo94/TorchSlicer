"""A batch's inputs: one tensor, or a tuple / list / dict of them (nested), as
``forward(*args)`` or ``forward(**kwargs)`` takes them.

Every executor moves them the same way: the *main* tensor (``"input_ids"`` if the batch has
one, else the first) is the slice's input, the others travel next to it as named tensors
(``aux``) to slice 0 only. A traced graph of a model with several inputs starts with one
``input.<name>`` node per tensor (``_Input``), which reads it from there; a later slice
gets an input it needs across the cuts like any other value (``core/multicut.py``).

Names come from the batch's structure: a dict's keys, a tuple's positions (``"0"``,
``"1"``), joined by dots when nested (``"images.0"``). ``(tensor, {name: tensor})`` is the
main tensor and named ones (``AuxInputStage``'s convention).
"""
import inspect

import torch
import torch.nn as nn
from torch.utils import _pytree as pytree


def _key(k) -> str:
    for attr in ("key", "idx", "name"):
        if hasattr(k, attr):
            return str(getattr(k, attr))
    return str(k)


def _named(tree) -> list:
    flat, _ = pytree.tree_flatten_with_path(tree)
    return [(".".join(_key(k) for k in path) or "input", v) for path, v in flat
            if isinstance(v, torch.Tensor)]


def _main_and_named(inputs) -> bool:
    return (isinstance(inputs, (tuple, list)) and len(inputs) == 2
            and isinstance(inputs[0], torch.Tensor) and isinstance(inputs[1], dict))


def input_leaves(inputs) -> list:
    """``[(name, tensor), ...]``: the batch's tensors, in order."""
    if isinstance(inputs, torch.Tensor):
        return [("input", inputs)]
    if _main_and_named(inputs):
        return [("0", inputs[0])] + _named(inputs[1])
    return _named(inputs)


def main_name(names: list) -> str:
    return "input_ids" if "input_ids" in names else names[0]


def split_inputs(inputs):
    """``(main, {name: tensor})``: what an executor sends to slice 0."""
    if isinstance(inputs, torch.Tensor):
        return inputs, {}
    leaves = input_leaves(inputs)
    if not leaves:
        raise TypeError(f"the batch's inputs hold no tensor: {type(inputs).__name__}")
    main = main_name([n for n, _ in leaves])
    return dict(leaves)[main], {n: t for n, t in leaves if n != main}


def count(inputs) -> int:
    """How many tensors the batch's inputs hold."""
    return 1 if isinstance(inputs, torch.Tensor) else len(input_leaves(inputs))


def bind(network: nn.Module, inputs):
    """``(args, kwargs)`` of ``network``'s forward for the batch's inputs: a tensor or a
    tuple / list are positional, a dict whose keys are all forward's parameters is keyword
    arguments (else one positional dict), ``(tensor, {name: tensor})`` both."""
    if isinstance(inputs, torch.Tensor):
        return (inputs,), {}
    params = inspect.signature(network.forward).parameters
    var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())

    def keywords(d) -> bool:
        return all(isinstance(k, str) for k in d) and (var_kw or set(d) <= set(params))

    if isinstance(inputs, dict):
        return ((), dict(inputs)) if keywords(inputs) else ((inputs,), {})
    if _main_and_named(inputs) and keywords(inputs[1]):
        return (inputs[0],), dict(inputs[1])
    if isinstance(inputs, (tuple, list)):
        return tuple(inputs), {}
    return (inputs,), {}


class _Input(nn.Module):
    """One input of a model with several: the main tensor, or a named one from the batch."""

    accepts_aux_inputs = True

    def __init__(self, name: str, main: bool):
        super().__init__()
        self.name, self.main = name, main

    def forward(self, x, **aux):
        if self.main:
            return x
        if self.name not in aux:
            raise KeyError(f"the batch has no input {self.name!r} (it has the main input and "
                           f"{sorted(aux)}); the graph was traced with it")
        return aux[self.name]

    def extra_repr(self):
        return repr(self.name) + (", main" if self.main else "")


# ── recorded for restore: what the example input looked like ──────────────────

_SMALL = 4096          # integer / bool tensors up to this size are recorded with their values


def describe(inputs) -> dict:
    """A JSON-able description of the batch's inputs, from which ``example`` rebuilds one
    (float tensors as zeros; integer and bool ones with their values when small, since
    ids or masks may decide what a model does)."""
    if isinstance(inputs, torch.Tensor):
        out = {"shape": list(inputs.shape), "dtype": str(inputs.dtype).removeprefix("torch.")}
        if not inputs.is_floating_point() and inputs.numel() <= _SMALL:
            out["values"] = inputs.detach().cpu().flatten().tolist()
        return out
    if isinstance(inputs, dict):
        return {"dict": [[k, describe(v)] for k, v in inputs.items()]}
    if isinstance(inputs, (tuple, list)):
        return {type(inputs).__name__: [describe(v) for v in inputs]}
    if inputs is None or isinstance(inputs, (bool, int, float, str)):
        return {"value": inputs}
    raise TypeError(f"can't record an input of type {type(inputs).__name__}")


def example(desc: dict):
    """A batch's inputs as ``describe`` recorded them."""
    if "dict" in desc:
        return {k: example(v) for k, v in desc["dict"]}
    if "tuple" in desc:
        return tuple(example(v) for v in desc["tuple"])
    if "list" in desc:
        return [example(v) for v in desc["list"]]
    if "value" in desc:
        return desc["value"]
    dtype = getattr(torch, desc["dtype"])
    if "values" in desc:
        return torch.tensor(desc["values"], dtype=dtype).reshape(desc["shape"])
    return torch.zeros(desc["shape"], dtype=dtype)
