import numpy as np


class Loss:
    @staticmethod
    def mean_squared_error(y_true, y_pred, derivative=False):
        if derivative:
            return y_pred - y_true
        return np.mean(np.power(y_true - y_pred, 2))
