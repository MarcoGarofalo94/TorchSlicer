#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Marco Garofalo

import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import Union

class SplitLayer(nn.Module):

    def __init__(self, layers: list, is_last: bool = False):
        """
        SplitLayer gets a list or Neural Network (sub)list of layers and expose methods to train  them.

        Parameters:
        layers (list): the list of layers to train. Can be a portion of the network but also the entire network.

        is_last (bool): indicates if the considered layer is the last portion of the network. This is needed for the backpropagation step. Note that is you pass a full network you should set it to True.
        """
        super(SplitLayer, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.x = None
        self.optimizer = None
        self._is_last = is_last

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """
        This method set the optimizer used after the backward step, e.g. SGD, Adam etc.
        """
        self.optimizer = optimizer

    def forward(self, input: torch.Tensor):
        """ Forward Step
            - 
            This method peforms the forward step of the layers passed as input. It uses the Variable class to register the input as a variable that requires the gradient.

            Parameters
            -
            input (Tensor): input of the the layers

            Return
            -
            input (Tensor): the output of the forward pass 
        """
        self.x = Variable(input.data, requires_grad=True)
        input = self.x
        for layer in self.layers:
            input = layer(input)
        return input

    def backward(self, prev_g: Union[torch.Tensor, None] = None, loss: Union[torch.Tensor, None] = None, out: Union[torch.Tensor, None] = None):
        """ Backward Step
            -
            This method computes the gradient of the output of the layer w.r.t. the input provided.

            Parameters
            -
            prev_g (Tensor): gradient of subsequent layers. If is last layer prev_g is None.
            Else you need to pass the gradient of subsequent layer computed previously

            loss (Tensor): loss computed after the forward step, needed if it is the last layer.

            out (Tensor): layer output from which to calculate the gradient w.r.t. the layer input, taking into account the gradient of previous layers (e.g. prev_g)

            Return
            -
            self.x.grad (Tensor): gradient of the layer
        """
        if self._is_last:
            loss.backward()
        else:
            out.backward( gradient=prev_g)
        # self.x.backward(retain_graph=True,gradient=g)
        return self.x.grad

    def optimize(self, dry_run: bool = False):
        """ Gradient Descent 
            -
            This method performs the gradient descent step, e.g. update weights with the optimizer previously set with the self.set_optimizer() method.

            Parameters
            -

            dry_run (bool): If True the weights of the network will not change (useful for debug)
        """
        assert self.optimizer != None
        if not dry_run:
            self.optimizer.step()
        self.optimizer.zero_grad()

