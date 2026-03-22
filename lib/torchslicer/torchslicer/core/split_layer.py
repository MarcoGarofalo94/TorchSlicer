#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Marco Garofalo

import torch
import torch.nn as nn
from typing import Union


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
        Empty list → layers[i] takes the partition's external input.
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
        self.x = None
        self.optimizer = None

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Wrap external input so we can retrieve its gradient at the cut point
        if input.is_floating_point():
            self.x = input.detach().requires_grad_(True)
            x_in = self.x
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
                for layer in self.layers:
                    current = layer(current)
                return current

            # DAG execution: track each layer's output, route inputs accordingly
            outputs: dict[int, torch.Tensor] = {}
            for i, layer in enumerate(self.layers):
                local_preds = preds[i]
                if not local_preds:
                    inp = x_in
                elif len(local_preds) == 1:
                    inp = outputs[local_preds[0]]
                else:
                    # Multi-input layer (e.g. _AddWrapper for skip connections)
                    inputs = [outputs[p] for p in local_preds]
                    outputs[i] = layer(*inputs)
                    continue
                outputs[i] = layer(inp)

        return outputs[len(self.layers) - 1]

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
        assert self.optimizer is not None, "Call set_optimizer() before optimize()"
        if not dry_run:
            self.optimizer.step()
        self.optimizer.zero_grad()
