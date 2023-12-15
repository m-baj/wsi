from dataclasses import dataclass
import numpy as np
from random import uniform


@dataclass
class SVM:
    C: float = 0.3
    kernel: str = "linear"
    sigma: float = 0.1

    def __post_init__(self):
        if self.kernel == "linear":
            self.kernel = self._linear
        elif self.kernel == "rbf":
            self.kernel = self._rbf

        self.w = None
        self.b = 0
        self.X = None
        self.y = None

    def fit(self, X, Y, learning_rate=1e-5, iterations=1000):
        samples = X.shape[0]
        self.w = np.array([uniform(0, 1) for _ in range(samples)])
        self.X = X
        self.y = Y
        krn = self.kernel(X, X)
        losses = []
        for _ in range(iterations):
            current_status = Y * (self.w.dot(krn) + self.b)
            missclassified = np.where(current_status < 1)[0]
            self.w -= learning_rate * self._w_gradient(missclassified, krn)
            self.b -= learning_rate * self._b_gradient(missclassified)
            losses.append(self._hinge_loss(krn))

        return losses

    def predict(self, X):
        return np.sign(self.w.dot(self.kernel(self.X, X)) + self.b)

    def _linear(self, x, y):
        return x.dot(y.T)

    def _rbf(self, x, y):
        return np.exp(
            -self.sigma
            * np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], axis=2) ** 2
        )
    
    def _poly(self, x, y, d):
        return (1 + x.T.dot(y)) ** d

    def _hinge_loss(self, krn):
        loss = 0.5 * self.w.dot(krn.dot(self.w)) + self.C * np.sum(
            np.maximum(0, 1 - self.y * (self.w.T.dot(krn) + self.b))
        )
        return loss

    def _w_gradient(self, indexes, krn):
        grad = krn.dot(self.w) - self.C * self.y[indexes].dot(krn[indexes])
        return grad

    def _b_gradient(self, indexes):
        grad = -self.C * np.sum(self.y[indexes])
        return grad
