#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Marco Garofalo

import torch
import torch.nn as nn
from typing import Union


def _is_inplace(module: nn.Module) -> bool:
    """``nn.ReLU(inplace=True)`` and friends, or a wrapped in-place op such as ``x.relu_()``."""
    if getattr(module, "inplace", False) is True:
        return True
    target = getattr(module, "target", None) if type(module).__name__ == "_FunctionWrapper" else None
    name = target if isinstance(target, str) else getattr(target, "__name__", "")
    return name.endswith("_") and not name.endswith("__")


class SplitLayer(nn.Module):
    """
    Wraps a contiguous partition of layers.  Owns forward, backward, and
    optimize for that slice.

    Parameters
    ----------
    layers : list[nn.Module]
        The layers in this partition.
    is_last : bool
        True for the final partition (uses loss.backward instead of
        out.backward(gradient=prev_g)).
    predecessors : list[list[int]] or None
        Intra-partition DAG info.  predecessors[i] is a list of
        partition-local indices of layers whose outputs feed into layers[i].
        Empty list → layers[i] takes the partition's external input; in a
        multi-input list, -1 stands for that external input.
        None (default) → purely sequential (equivalent to [[],[1],[2],...]).
    """

    def __init__(
        self,
        layers: list,
        is_last: bool = False,
        predecessors: list | None = None,
        mixed_precision: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self._is_last = is_last
        # Store predecessor info; None means sequential shortcut
        self._predecessors: list[list[int]] | None = predecessors
        # When True, forward runs inside torch.autocast (bfloat16).
        # bfloat16 does not require a GradScaler — recommended over float16.
        self._mixed_precision = mixed_precision
        self._amp_device = "cuda" if torch.cuda.is_available() else "cpu"
        # The cut-point input is a leaf that requires grad; an in-place op on it is an
        # autograd error, so layers reading it in place get a copy instead.
        if predecessors is None or all(len(p) == 0 for p in predecessors):
            takes_x = [0] if layers else []        # sequential fast path
        else:
            takes_x = [i for i, p in enumerate(predecessors) if not p or -1 in p]
        self._clone_input = any(_is_inplace(layers[i]) for i in takes_x)
        self.x = None
        self.optimizer = None
        self._optimizer_set = False

    def set_optimizer(self, optimizer: "torch.optim.Optimizer | None"):
        """``None`` = this slice has nothing to optimize (no parameters, or only tied
        parameters owned by another slice)."""
        self.optimizer = optimizer
        self._optimizer_set = True

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, input: torch.Tensor, **aux) -> torch.Tensor:
        """
        Parameters
        ----------
        input : torch.Tensor
            The main activation (or int64 token ids for the first partition).
        **aux :
            Optional extra named tensors for the first layer only.  Used by
            ``AuxInputStage`` subclasses (e.g. LLaVA vision encoder) that need
            ``pixel_values`` or other non-activation inputs.  Ignored by layers
            that do not declare ``accepts_aux_inputs = True``.
        """
        # Wrap external input so we can retrieve its gradient at the cut point
        if input.is_floating_point():
            self.x = input.detach().requires_grad_(True)
            x_in = self.x.clone() if self._clone_input else self.x
        else:
            # Integer input (e.g. token ids) — grad hooks on input not needed
            self.x = None
            x_in = input

        ctx = (
            torch.autocast(device_type=self._amp_device, dtype=torch.bfloat16)
            if self._mixed_precision
            else torch.autocast(device_type=self._amp_device, enabled=False)
        )
        with ctx:
            preds = self._predecessors
            if preds is None or all(len(p) == 0 for p in preds):
                # Fast path: purely sequential
                current = x_in
                for i, layer in enumerate(self.layers):
                    if i == 0 and aux and getattr(layer, 'accepts_aux_inputs', False):
                        current = layer(current, **aux)
                    else:
                        current = layer(current)
                return current

            # DAG execution: track each layer's output, route inputs accordingly
            outputs: dict[int, torch.Tensor] = {}
            for i, layer in enumerate(self.layers):
                local_preds = preds[i]
                if not local_preds:
                    inp = x_in
                    if aux and getattr(layer, 'accepts_aux_inputs', False):
                        outputs[i] = layer(inp, **aux)
                        continue
                elif len(local_preds) == 1:
                    inp = outputs[local_preds[0]]
                else:
                    # Multi-input layer (e.g. _AddWrapper for skip connections);
                    # -1 = this partition's external input
                    layer_inputs = [outputs[p] if p >= 0 else x_in for p in local_preds]
                    outputs[i] = layer(*layer_inputs)
                    continue
                outputs[i] = layer(inp)

        return outputs[len(self.layers) - 1]

    def pop_moe_aux_loss(self) -> "torch.Tensor | None":
        """Sum and clear accumulated MoE aux losses from all MoEBlockStage in this partition.

        Called after each backward pass so MoE router weights receive a separate
        gradient that does not travel across partition boundaries.
        """
        total = None
        for layer in self.layers:
            if hasattr(layer, 'pop_aux_loss'):
                aux = layer.pop_aux_loss()
                if aux is not None:
                    total = aux if total is None else total + aux
        return total

    # ── backward ──────────────────────────────────────────────────────────────

    def backward(
        self,
        prev_g: Union[torch.Tensor, None] = None,
        loss: Union[torch.Tensor, None] = None,
        out: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor | None:
        if self._is_last:
            loss.backward()
        else:
            out.backward(gradient=prev_g)
        return self.x.grad if self.x is not None else None

    # ── optimize ──────────────────────────────────────────────────────────────

    def optimize(self, dry_run: bool = False):
        assert self._optimizer_set, "Call set_optimizer() before optimize()"
        if self.optimizer is not None:
            if not dry_run:
                self.optimizer.step()
            self.optimizer.zero_grad()
        self.x = None  # release cut-point tensor; prevents CUDA allocator growth
