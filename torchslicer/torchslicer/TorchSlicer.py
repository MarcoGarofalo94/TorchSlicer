#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Marco Garofalo


from typing import Union
import inspect
import re
from torch import Module
import json


class Slicer():
    """
    This class gets a network, slices it and returns the configuration of it.
    """

    def __init__(self, network: Union[Module, list[Module]]):
        self.network = network
        self.layers = self._get_layers(network=network)
        self.config = self.get_config()

    def _get_layers(self, network):
        """
        This method gets a network and returns the layers of it.
        """
        return [layer for layer in network.children()]

    def _parse_layers(self, layers):
        """
        This method gets a list of layers and returns the layers of it.
        """
        layers_info = []
        for layer in layers:
            layer_name = layer.__class__.__name__
            layer_info = {"layer": layer_name}

            # Get constructor parameters
            constructor_params = inspect.signature(layer.__init__).parameters

            # Extract specified parameters and their values
            for param_name, param in constructor_params.items():
                if param_name != "self" and param_name in layer.__dict__:
                    layer_info[param_name] = layer.__dict__[param_name]

            # Extract additional parameters from the layer string using regex
            layer_string = str(layer)
            additional_params = re.findall(
                r'(\w+)\s*=\s*([^,\s)]+)', layer_string)

            # Add additional parameters to layer_info
            for param_name, param_value in additional_params:
                if param_name not in layer_info:
                    # Convert bias parameter to boolean if it's 'True' or 'False' string
                    param_value = eval(param_value)
                    layer_info[param_name] = param_value
            if hasattr(layer, 'weight'):
                layer_info['dtype'] = str(layer.weight.dtype)

            layers_info.append(layer_info)
        return layers_info

    def get_config(self, save_to_file: str = ""):
        """
        This method gets a network and returns the configuration of it.
        """

        if save_to_file != "":
            with open(save_to_file, 'w') as f:
                json.dump(self.config, f)
        return self._parse_layers(self._parse_layers(self.layers))
