import numpy as np

DEFAULT_LEARNING_RATE = 0.1
DEFAULT_EPSILON = 1e-6
DEFAULT_MAX_ITER = 1000


# class Solver:
#     def __init__(self, function, grad, x0, learning_rate, epsilon, max_iter):
#         self.function = function
#         self.grad = grad
#         self.x0 = x0
#         self.learning_rate = learning_rate
#         self.epsilon = epsilon
#         self.max_iter = max_iter
#         self.minimum = None
#         self.all_steps = np.array([x0])

#     def get_minimum(self):
#         if self.minimum is None:
#             self._find_min()
#         return self.minimum

#     def _perform_gradient_descent(self):
#         x = self.x0
#         for i in range(self.max_iter):
#             x = x - self.learning_rate * self.grad(*x)
#             if np.linalg.norm(self.grad(*x)) <= self.epsilon:
#                 break
#             self.all_steps = np.append(self.all_steps, [x], axis=0)
#         self.minimum = x


def gradient_descent(
    f,
    grad,
    x0,
    learning_rate=DEFAULT_LEARNING_RATE,
    epsilon=DEFAULT_EPSILON,
    max_iter=DEFAULT_MAX_ITER,
):
    """
    f: callable - function to minimize
    grad: callable - gradient of f
    x0: np.array - initial point
    learning_rate: float - learning rate
    epsilon: float - precision
    max_iter: int - maximum number of iterations
    """
    all_points = np.array([x0])
    iterations = 0
    x = x0
    for i in range(max_iter):
        x = x - learning_rate * grad(*x)
        iterations += 1
        if np.linalg.norm(grad(*x)) <= epsilon:
            break
        all_points = np.append(all_points, [x], axis=0)
    return x, all_points, iterations
