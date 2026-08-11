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
    def __init__(self, layer_dims : list, activation="relu"):
        self.layer_dims = layer_dims
        self.layer_count = len(layer_dims)

        if activation=="relu":
            self._activation = lambda x: x * (x > 0)
            self._d_activation = lambda x: (x > 0)
        else:
            self._activation = lambda x: x
            self._d_activation = lambda: 1

        self._x = [np.zeros([]) for i in self.layer_dims] # list of layers before activation
        self._z = [np.zeros([]) for i in self.layer_dims] # list of layers after activation

        # randomize all weights and biases
        # TODO: Optimize choices for weights (may depend on acticvation)
        self._w = [np.random.rand(self.layer_dims[i + 1], self.layer_dims[i]) - 0.5 for i in range(self.layer_count - 1)]
        self._b = [np.zeros((self.layer_dims[i])) for i in range(self.layer_count - 1)]

    def forward_prop(self, input : list):
        return None

    def backward_prop(self, err : list):
        return None
