#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Marco Garofalo

import io
from torch import nn
from torch.nn.modules.module import Module
from typing import Union
import torch


class Slicer:
    """
    Inspects a sequential nn.Module and serialises each child layer so it can
    be shipped to a remote device and reinstantiated exactly.

    Serialisation uses torch.save() (PyTorch's own pickling), which captures:
      - the layer class and all constructor arguments (including bool, str,
        tuple, dtype, groups, dilation, padding_mode, custom attributes …)
      - the full state_dict (weights + buffers)

    This works for any nn.Module — built-in or custom — as long as the class
    is importable on the receiving end.
    """

    def __init__(self, network: Union[Module, list[Module]]):
        self.network = network
        self.layers: list[Module] = list(network.children())

    def get_config(self) -> list[dict]:
        """
        Return one entry per layer:
            {
                'layer_type': 'Linear',         # class name, for logging
                'serialized': <bytes>,           # torch.save(layer) bytes
            }
        """
        result = []
        for layer in self.layers:
            buf = io.BytesIO()
            torch.save(layer, buf)
            result.append({
                'layer_type': layer.__class__.__name__,
                'serialized': buf.getvalue(),
            })
        return result

    @staticmethod
    def load_layer(config: dict) -> nn.Module:
        """Deserialise a single layer config produced by get_config()."""
        return torch.load(io.BytesIO(config['serialized']), weights_only=False)

    @staticmethod
    def load_layers(configs: list[dict]) -> list[nn.Module]:
        """Deserialise a list of layer configs produced by get_config()."""
        return [Slicer.load_layer(c) for c in configs]
