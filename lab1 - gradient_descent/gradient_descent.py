import numpy as np

DEFAULT_LEARNING_RATE = 0.1
DEFAULT_EPSILON = 1e-6
DEFAULT_MAX_ITER = 1000


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
        gradient = grad(*x)
        x = x - learning_rate * gradient
        iterations += 1
        if np.linalg.norm(gradient) <= epsilon:
            break
        all_points = np.append(all_points, [x], axis=0)
    return x, all_points, iterations


x0 = np.array([1.0, 1.0])
f = lambda x, y: x**2 + y**2
grad = lambda x, y: np.array([2 * x, 2 * y])

x, all_points, iterations = gradient_descent(f, grad, x0)
print(x)
print(iterations)
