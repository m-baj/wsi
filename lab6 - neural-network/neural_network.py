from dataclasses import dataclass, field
import time
import numpy as np

from activation import Activation
from loss import Loss

from tqdm import tqdm


@dataclass
class InputLayer:
    units: int
    previous_layer = None
    next_layer = None
    outputs = None


@dataclass
class HiddenLayer(InputLayer):
    activation: Activation
    delta = None
    error = None
    weights = None
    biases = None

    def _init_weights(self):
        self.weights = np.random.uniform(
            -1 / np.sqrt(self.units),
            1 / np.sqrt(self.units),
            (self.units, self.previous_layer.units),
        )

    def _init_biases(self):
        self.biases = np.zeros((self.units, 1))

    def forward(self, inputs):
        outputs = []
        for i in range(self.units):
            outputs.append(
                self.activation(np.dot(self.weights[i], inputs) + self.biases[i])
            )
        self.outputs = np.array(outputs)
        return np.array(outputs)


@dataclass
class NeuralNetwork:
    layers: np.array
    loss: Loss = field(default=Loss.mean_squared_error)

    def __post_init__(self):
        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        for i, layer in enumerate(self.layers):
            layer.previous_layer = self.layers[i - 1] if i > 0 else None
            layer.next_layer = self.layers[i + 1] if i < len(self.layers) - 1 else None

    def _init_weights(self):
        for layer in self.layers[1:]:
            layer._init_weights()
            layer._init_biases()

    def _forward(self, inputs):
        if len(inputs) != self.layers[0].units:
            raise ValueError(
                f"Input layer expects {self.layers[0].units} inputs, but got {len(inputs)}"
            )
        curr_outputs = inputs
        for layer in self.layers[1:]:
            curr_outputs = layer.forward(curr_outputs)
        return curr_outputs

    def fit(self, X_train, y_train, epochs, learning_rate):
        for epoch in tqdm(range(epochs)):
            error = 0
            for X, y in zip(X_train, y_train):
                self.layers[0].outputs = X.reshape(len(X), 1)
                y_pred = self._forward(X)
                error += self.loss(y, y_pred)
                self._backward(y, y_pred, learning_rate)

    def predict(self, X_test):
        y_pred = []
        for X in X_test:
            res = self._forward(X)
            res = res.reshape(len(res))
            y_pred.append(res)

        return convert_to_one_hot(np.array(y_pred))

    def _backward(self, y, y_pred, learning_rate):
        y = y.reshape(len(y), 1)
        for layer in reversed(self.layers[1:]):
            if layer == self.layers[-1]:
                layer.delta = self.loss(y, y_pred, derivative=True) * layer.activation(
                    y_pred, derivative=True
                )
            else:
                layer.error = np.dot(layer.next_layer.weights.T, layer.next_layer.delta)
                layer.delta = layer.error * layer.activation(
                    layer.outputs, derivative=True
                )

            dc_db = layer.delta
            dc_dw = np.matmul(layer.delta, layer.previous_layer.outputs.T)
            layer.biases -= learning_rate * dc_db
            layer.weights -= learning_rate * dc_dw


def convert_to_one_hot(Y):
    one_hot = []
    for y in Y:
        i = np.argmax(y)
        vector = np.zeros(len(y))
        vector[i] = 1
        one_hot.append(vector)

    return np.array(one_hot)


def confusion_matrix(y_test, y_pred):
    matrix = np.zeros((len(y_test[0]), len(y_test[0])))
    for i, y in enumerate(y_test):
        matrix[np.argmax(y)][np.argmax(y_pred[i])] += 1
    return matrix


def precision_score(y_test, y_pred, average="micro"):
    matrix = confusion_matrix(y_test, y_pred)
    if average == "micro":
        return np.trace(matrix) / np.sum(matrix)
    elif average == "macro":
        return np.mean([matrix[i][i] / np.sum(matrix[i]) for i in range(len(matrix))])
    else:
        raise ValueError(
            f"Average must be either 'micro' or 'macro', but got {average}"
        )
