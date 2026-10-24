"""Deterministic GPU kernels (``training.deterministic``).

cuDNN's default convolution backward and some cuBLAS paths use atomics, so two GPU runs from
the same seed drift apart (on a small CNN, by ~0.1 in loss within 3 epochs). CPU runs are
deterministic already. With the flag on, every process of a run (the data owner, each worker,
each client node) calls ``set_deterministic()``: same seed, same losses, at some speed cost.
"""
import os
import warnings

import torch


def set_deterministic() -> None:
    """cuDNN deterministic algorithms, no autotuning, deterministic torch ops where they exist
    (others warn instead of raising). ``CUBLAS_WORKSPACE_CONFIG`` takes effect only if set
    before this process first uses cuBLAS, so call this before the model touches the GPU."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.use_deterministic_algorithms(True, warn_only=True)


def _generators(loader) -> list:
    """The ``torch.Generator``s a loader (or each of a session's ``loaders``) shuffles with."""
    out = []
    for ld in getattr(loader, "loaders", None) or [loader]:
        sampler = getattr(ld, "sampler", None)
        batch_sampler = getattr(ld, "batch_sampler", None)
        for g in (getattr(ld, "generator", None), getattr(sampler, "generator", None),
                  getattr(getattr(batch_sampler, "sampler", None), "generator", None)):
            if isinstance(g, torch.Generator) and all(g is not o for o in out):
                out.append(g)
    return out


def rng_snapshot(loader) -> dict:
    """The random state an epoch over ``loader`` draws from: the loader's shuffle generators
    and the process's CPU / CUDA generators (dropout, a loader without its own generator)."""
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
            "loader": [g.get_state() for g in _generators(loader)]}


def rng_restore(snap: dict, loader) -> None:
    """Put back an ``rng_snapshot``: an epoch retried after recovery then sees the same
    batches in the same order (and the same dropout masks) as its first attempt."""
    torch.set_rng_state(snap["cpu"])
    if snap["cuda"] is not None:
        torch.cuda.set_rng_state_all(snap["cuda"])
    for g, state in zip(_generators(loader), snap["loader"]):
        g.set_state(state)
