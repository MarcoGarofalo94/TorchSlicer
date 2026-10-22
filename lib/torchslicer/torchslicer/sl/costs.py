"""Hardware-independent costs of each slice: memory and FLOPs (in-process runs).

Measured once, on the run's first batch, by running each slice forward on it (forked RNG,
buffers restored, no backward), so training is unchanged:

- ``param_bytes``: parameters and buffers; ``grad_bytes``: gradients of the trainable ones;
  ``optimizer_bytes``: the optimizer's state (state tensors per parameter x ``grad_bytes``);
- ``activation_bytes``: what the slice saves for backward on one batch, parameters excluded
  (``torch.autograd.graph.saved_tensors_hooks``), i.e. at that batch size;
- ``fwd_flops``: forward FLOPs on one batch (``torch.utils.flop_counter``); a training step
  is counted as 3x forward (forward + backward ~ 2x forward), the usual estimate.

These depend on the model, the batch shape, the dtype and the optimizer, not on the device
or the network.
"""
from __future__ import annotations

import torch

_TRAIN_PER_FWD = 3


def _nbytes(t) -> int:
    return t.numel() * t.element_size() if isinstance(t, torch.Tensor) else 0


def _state_tensors(optimizer_cfg: dict | None) -> int:
    if not optimizer_cfg:
        return 0
    from ..offload.executor import optimizer_state_tensors
    try:
        return optimizer_state_tensors(optimizer_cfg)
    except Exception:
        return 0


def slice_costs(split_layers: list, inputs: torch.Tensor, optimizer_cfg: dict | None = None,
                aux: dict | None = None) -> list[dict]:
    """Per slice: ``param_bytes, grad_bytes, optimizer_bytes, activation_bytes, fwd_flops``
    for one batch of ``inputs`` (see the module docstring)."""
    from torch.utils.flop_counter import FlopCounterMode
    k = _state_tensors(optimizer_cfg)
    saved = [(b, b.detach().clone()) for sl in split_layers for b in sl.buffers()]
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    out, x = [], inputs
    with torch.random.fork_rng(devices=devices):
        for i, sl in enumerate(split_layers):
            params = list({id(p): p for p in sl.parameters()}.values())
            ptrs = {p.data_ptr() for p in params}
            trainable = sum(_nbytes(p) for p in params if p.requires_grad)
            acts = [0]

            def pack(t):
                if t.data_ptr() not in ptrs:
                    acts[0] += _nbytes(t)
                return t

            xin = x
            if isinstance(x, torch.Tensor) and x.is_floating_point():
                xin = x.detach().requires_grad_(True).clone()     # as SplitLayer.forward: in-place ops ok
            counter = FlopCounterMode(display=False)
            with torch.enable_grad(), counter, torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                y = sl.run(xin, **(aux or {})) if i == 0 else sl.run(xin)
            out.append({"param_bytes": sum(_nbytes(p) for p in params) + sum(_nbytes(b) for b in sl.buffers()),
                        "grad_bytes": trainable, "optimizer_bytes": k * trainable,
                        "activation_bytes": acts[0], "fwd_flops": int(counter.get_total_flops())})
            x = y.detach() if isinstance(y, torch.Tensor) else y
    with torch.no_grad():
        for b, v in saved:
            b.copy_(v)
    return out


def cost_fields(costs: list[dict], batch_size: int, samples: int, client_slices) -> dict:
    """The epoch's history fields from ``slice_costs`` measured at ``batch_size``:
    ``mem_bytes`` (per slice: params + grads + optimizer state + activations of one batch),
    ``flops`` (per slice: training FLOPs over the epoch's ``samples``, scaled per sample),
    and their client-side sums ``client_mem_bytes`` / ``client_flops``."""
    mem = [c["param_bytes"] + c["grad_bytes"] + c["optimizer_bytes"] + c["activation_bytes"] for c in costs]
    per_sample = [_TRAIN_PER_FWD * c["fwd_flops"] / batch_size for c in costs]
    flops = [int(round(f * samples)) for f in per_sample]
    client = set(client_slices)
    return {"mem_bytes": mem, "flops": flops,
            "client_mem_bytes": sum(m for i, m in enumerate(mem) if i in client),
            "client_flops": sum(f for i, f in enumerate(flops) if i in client),
            "costs_batch_size": batch_size}
