"""
mlp.py

A from-scratch implementation of a Multilayer Perceptron (MLP).

Supports configurable layer sizes, forward propagation, and (optionally)
backpropagation for training via gradient descent.

Author:         Eli DeWitt
Created:        2026-08-10
"""

import numpy as np

class MLP:
    def __init__(self, layer_dims : list):
        self.layer_dims = layer_dims

    def forward_prop(self, input : list):
        return None

    def backward_prop(self, err : list):
        return None
