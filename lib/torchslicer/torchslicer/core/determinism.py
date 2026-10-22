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
