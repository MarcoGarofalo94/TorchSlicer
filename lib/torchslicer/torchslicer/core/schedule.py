"""Learning-rate schedules: a factor of each param group's base learning rate per optimizer step.

``training.scheduler: {name: linear, params: {warmup_steps: 500, total_steps: 26290}}`` makes
every slice set ``lr = base_lr * factor(t)`` right before its ``t``-th optimizer step (``t``
from 0), as ``torch.optim.lr_scheduler.LambdaLR`` would. Writing the lr at each step keeps it
right when ``ts.sl`` sessions swap optimizer states between clients; ``t`` counts the slice's
steps over all clients. Local executor only for now.
"""
from __future__ import annotations

from typing import Callable


def linear(warmup_steps: int = 0, total_steps: int | None = None) -> Callable[[int], float]:
    """Linear warmup from 0, then linear decay to 0 at ``total_steps`` (constant without it);
    the same factors as Hugging Face's ``get_linear_schedule_with_warmup``."""
    def factor(t: int) -> float:
        if t < warmup_steps:
            return t / max(1, warmup_steps)
        if total_steps is None:
            return 1.0
        return max(0.0, (total_steps - t) / max(1, total_steps - warmup_steps))
    return factor


_SCHEDULES = {"linear": linear}


def register_schedule(name: str, fn) -> None:
    """``fn(**params) -> factor(step) -> float``."""
    _SCHEDULES[name] = fn


def build_schedule(cfg: dict | None) -> Callable[[int], float] | None:
    if not cfg:
        return None
    name = cfg["name"]
    if name not in _SCHEDULES:
        raise KeyError(f"unknown scheduler {name!r}; available: {sorted(_SCHEDULES)}")
    return _SCHEDULES[name](**(cfg.get("params") or {}))
