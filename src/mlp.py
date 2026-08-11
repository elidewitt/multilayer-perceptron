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
    def __init__(self, layer_dims : list, activation="relu", final_activation="softmax"):
        self.layer_dims = layer_dims
        self.layer_count = len(layer_dims)

        if activation=="relu":
            self._activation = self._relu
            self._d_activation = self._d_relu
        else:
            self._activation = self._identity
            self._d_activation = self._d_identity

        if final_activation=="softmax":
            self._final_activation = self._softmax
            self._d_final_activation = self._d_softmax

        self._x = [np.zeros([]) for i in self.layer_dims] # list of layers before activation
        self._z = [np.zeros([]) for i in self.layer_dims] # list of layers after activation

        # randomize all weights and biases
        # TODO: Optimize choices for weights (may depend on acticvation)
        self._w = [np.random.rand(self.layer_dims[i + 1], self.layer_dims[i]) - 0.5 for i in range(self.layer_count - 1)]
        self._b = [np.zeros((self.layer_dims[i + 1])) for i in range(self.layer_count - 1)]

    # ---- activation functions ----

    def _relu(self, x):
        return x * (x > 0)

    def _d_relu(self, x, g):
        return (x > 0) * g

    def _identity(self, x):
        return x

    def _d_identity(self, x, g):
        return g

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _d_sigmoid(self, x, g):
        s = self._sigmoid(x)
        return s * (1 - s) * g

    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def _d_softmax(self, x, g):
        s = self._softmax(x)
        return s * g - s * np.dot(s, g)

    # ---- forward prop algorithm ----

    def forward_prop(self, input : list):

        self._x[0] = np.array(input)

        for i in range(self.layer_count - 1):
            self._z[i] = self._activation(self._x[i])
            self._x[i+1] = self._w[i] @ self._z[i] + self._b[i]

        self._z[-1] = self._final_activation(self._x[-1])
        return self._z[-1]


    def backward_prop(self, err : list):
        return None
