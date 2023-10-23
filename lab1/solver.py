import time
import numpy as np
from gradient_descent import gradient_descent


LEARNING_RATES = [0.001, 0.01, 0.1, 0.5, 1]
EPSILON = 1e-6


class Function:
    def __init__(self, f, grad, dim, domain):
        self.f = f
        self.grad = grad
        self.dim = dim
        self.domain = domain


class Experiment:
    def __init__(self, f, start_point, learning_rate, epsilon, max_iter):
        self.function = f
        self.start_point = start_point
        self.l_rate = learning_rate
        self.eps = epsilon
        self.max_iter = max_iter

    def conduct(self):
        dur, iters, path, min, is_min_found = self._find_minimum()
        result = ExpResult(self, dur, iters, path, min, is_min_found)
        return result

    def _find_minimum(self):
        time_start = time.time()
        minimum, path_to_min, iterations = gradient_descent(
            self.function.f,
            self.function.grad,
            self.start_point,
            self.l_rate,
            self.eps,
            self.max_iter,
        )
        time_end = time.time()
        dur = time_end - time_start
        is_min_found = np.linalg.norm(self.function.grad(*minimum)) <= self.eps
        return dur, iterations, path_to_min, minimum, is_min_found


class ExpResult:
    def __init__(
        self, experiment, duration, iterations, path_to_min, minimum, is_min_found
    ):
        self.experiment = experiment
        self.duration = duration
        self.iterations = iterations
        self.path_to_min = path_to_min
        self.minimum = minimum
        self.is_min_found = is_min_found


class Solver:
    def __init__(self, iterations_number, f, learning_rates):
        self.iterations_number = iterations_number
        self.f = f
        self.l_rates = learning_rates
        self.experiments = []
        self.results = []

    def get_experiments_number(self):
        return len(self.experiments)

    def generate_experiments(self, number_of_experiments):
        for _ in range(number_of_experiments):
            self._create_experiment(np.random.choice(self.l_rates), EPSILON)

    def solve(self):
        for exp in self.experiments:
            result = exp.conduct()
            self.results.append(result)

    def get_results_data(self):
        return [
            [
                res.experiment.l_rate,
                res.experiment.start_point,
                res.minimum,
                res.duration,
                res.iterations,
            ]
            for res in self.results
        ]

    def get_sorted_results_dict(self):
        return {
            l_rate: [res for res in self.results if res.experiment.l_rate == l_rate]
            for l_rate in self.l_rates
        }

    def _create_experiment(self, learning_rate, epsilon):
        start = self._get_random_point()
        new_exp = Experiment(
            self.f, start, learning_rate, epsilon, self.iterations_number
        )
        self.experiments.append(new_exp)

    def _get_random_point(self):
        left, right = self.f.domain
        return np.random.uniform(left, right, self.f.dim - 1)
