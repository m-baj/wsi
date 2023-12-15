from dataclasses import dataclass, field
from kernel_SVM import SVM
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
import numpy as np
from enum import Enum


@dataclass
class Result:
    C: float
    kernel: str
    sigma: float
    accuracy: float
    precision: float
    recall: float
    confusion_matrix: np.ndarray

    def __str__(self):
        return f"C: {self.C}, kernel: {self.kernel}, sigma: {self.sigma}, accuracy: {self.accuracy}, precision: {self.precision}, recall: {self.recall}, confusion_matrix: {self.confusion_matrix}"

    def __repr__(self):
        return self.__str__()


class Parameter(Enum):
    C = 1
    sigma = 2


@dataclass
class ExperimentConductor:
    train_X: np.ndarray
    train_Y: np.ndarray
    test_X: np.ndarray
    test_Y: np.ndarray
    kernel: str
    C_values: list = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    sigma_values: list = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])

    def tune_C(self, learning_rate=1e-5, iterations=1000):
        results = []
        for value in self.C_values:
            svm = SVM(kernel=self.kernel, C=value)
            svm.fit(
                self.train_X,
                self.train_Y,
                learning_rate=learning_rate,
                iterations=iterations,
            )
            y_predicted = svm.predict(self.test_X)
            accuracy = accuracy_score(self.test_Y, y_predicted)
            precision = precision_score(self.test_Y, y_predicted)
            recall = recall_score(self.test_Y, y_predicted)
            conf_matrix = confusion_matrix(self.test_Y, y_predicted)
            results.append(
                Result(
                    C=value,
                    kernel=self.kernel,
                    sigma=svm.sigma,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    confusion_matrix=conf_matrix,
                )
            )
        return results

    def tune_sigma(self, learning_rate=1e-5, iterations=1000):
        results = []
        for value in self.sigma_values:
            svm = SVM(kernel=self.kernel, sigma=value, C=0.7)
            svm.fit(
                self.train_X,
                self.train_Y,
                learning_rate=learning_rate,
                iterations=iterations,
            )
            y_predicted = svm.predict(self.test_X)
            accuracy = accuracy_score(self.test_Y, y_predicted)
            precision = precision_score(self.test_Y, y_predicted)
            recall = recall_score(self.test_Y, y_predicted)
            conf_matrix = confusion_matrix(self.test_Y, y_predicted)
            results.append(
                Result(
                    C=svm.C,
                    kernel=self.kernel,
                    sigma=value,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    confusion_matrix=conf_matrix,
                )
            )
        return results

    def conduct_experiment(
        self, number_of_experiments, learning_rate=1e-5, iterations=10000
    ):
        results = []
        for _ in range(number_of_experiments):
            svm = SVM(kernel=self.kernel, sigma=0.1, C=0.9)
            svm.fit(
                self.train_X,
                self.train_Y,
                learning_rate=learning_rate,
                iterations=iterations,
            )
            y_predicted = svm.predict(self.test_X)
            accuracy = accuracy_score(self.test_Y, y_predicted)
            precision = precision_score(self.test_Y, y_predicted)
            recall = recall_score(self.test_Y, y_predicted)
            conf_matrix = confusion_matrix(self.test_Y, y_predicted)
            results.append(
                Result(
                    C=svm.C,
                    kernel=self.kernel,
                    sigma=svm.sigma,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    confusion_matrix=conf_matrix,
                )
            )
        return results
