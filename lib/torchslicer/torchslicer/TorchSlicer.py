#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Marco Garofalo


import io
from typing import Union
import inspect
import re
from torch.nn.modules.module import Module
import json
import torch
import base64

class Slicer():
    """
    This class gets a network, slices it and returns the configuration of it.
    """

    def __init__(self, network: Union[Module, list[Module]]):
        self.network = network
        self.layers = self._get_layers(network=network)

    def _get_layers(self, network):
        """
        This method gets a network and returns the layers of it.
        """
        return list(network.children())

    def annotate_type(self, value):
        if isinstance(value, (list)):
            param_type = list(type(val).__name__ for val in value)
        elif isinstance(value, (tuple)):
            param_type = tuple(type(val).__name__ for val in value)
        else:
            param_type = type(value).__name__
        return param_type

    def _parse_layers(self, layers, init_state_dict: bool = False):
        layers_info = []
        for layer in layers:
            layer_name = layer.__class__.__name__
            layer_info = {"layer": layer_name, "attributes": {}}

            # Get constructor parameters
            constructor_params = inspect.signature(layer.__init__).parameters

            # Extract specified parameters and their values
            for param_name, param in constructor_params.items():
                if param_name != "self" and param_name in layer.__dict__:
                    layer_info["attributes"][param_name] = {
                        "value": layer.__dict__[param_name], "param_type": self.annotate_type(layer.__dict__[param_name])}

            # Extract additional parameters from the layer string using regex
            layer_string = str(layer)
            additional_params = re.findall(
                r'(\w+)\s*=\s*([^,\s)]+)', layer_string)

            # Add additional parameters to layer_info
            for param_name, param_value in additional_params:
                if param_name not in list(layer_info["attributes"].keys()):

                    param_value = eval(param_value)
                    layer_info["attributes"][param_name] = {
                        "value": param_value, "param_type": self.annotate_type(param_value)}

            if hasattr(layer, 'weight'):
                layer_info["attributes"]["dtype"] = {"value": str(
                    layer.weight.dtype), "param_type": self.annotate_type(layer.weight.dtype)}
                if init_state_dict:
                    buff = io.BytesIO()
                    torch.save(layer.state_dict(), buff)
                    state_dict = buff.getvalue()
                    layer_info["state_dict"] = base64.b64encode(state_dict).decode('utf-8')
                    buff.close()
                # layer_info["init_state_dict"] = layer.state_dict() if init_state_dict else b""
                
            layers_info.append(layer_info)
        return layers_info

    def get_config(self, save_to_file: str = "", init_state_dict: bool = False):
        """
        This method gets a network and returns the configuration of it.
        """
        if save_to_file != "":
            with open(save_to_file, 'w') as f:
                json.dump(self.config, f)
        return self._parse_layers(self.layers, init_state_dict=init_state_dict)

    @staticmethod
    def cast_layers(raw_layers: list):
        for i, layer in enumerate(raw_layers):
            raw_layers[i]['decoded_attributes'] = {}
            for attr, val in layer['attributes'].items():
                if (val['param_type'] not in ['float', 'dtype', "int"]) and (not isinstance(val['param_type'], (list, tuple))):
                    continue
                if val['param_type'] == 'dtype':
                    raw_layers[i]['attributes'][attr]['value'] = eval(
                        val["value"])
                elif isinstance(val['param_type'], list):
                    raw_layers[i]['attributes'][attr]['value'] = [
                        eval(val["param_type"][j]+"("+str(v)+")") for j, v in enumerate(val['value'])]
                else:
                    raw_layers[i]['attributes'][attr]['value'] = eval(
                        val["param_type"]+"("+str(val['value'])+")")
                raw_layers[i]['decoded_attributes'][attr] = raw_layers[i]['attributes'][attr]['value']
        return raw_layers
