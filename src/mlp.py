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
        self._b = [np.zeros((self.layer_dims[i + 1])) for i in range(self.layer_count - 1)]

    def forward_prop(self, input : list):

        self._x[0] = np.array(input)

        for i in range(self.layer_count - 1):
            self._z[i] = self._activation(self._x[i])
            self._x[i+1] = self._w[i] @ self._z[i] + self._b[i]

        # TODO: softmax last layer
        self._z[-1] = self._x[-1] 
        return self._z[-1]


    def backward_prop(self, err : list):
        return None
