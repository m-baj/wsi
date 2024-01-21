import numpy as np


class Activation:
    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative:
            return Activation.sigmoid(x) * (1 - Activation.sigmoid(x))
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x, derivative=False):
        if derivative:
            return np.where(x <= 0, 0, 1)
        return np.maximum(0, x)

    @staticmethod
    def tanh(x, derivative=False):
        if derivative:
            return 1 - np.tanh(x) ** 2
        return np.tanh(x)

    @staticmethod
    def linear(x, derivative=False):
        if derivative:
            return 1
        return x

    @staticmethod
    def leaky_relu(x, derivative=False):
        if derivative:
            return np.where(x <= 0, 0.01, 1)
        return np.maximum(0.01 * x, x)
