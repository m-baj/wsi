import numpy as np


def random_number_less_than(number: float) -> bool:
    return np.random.rand() <= number


def get_crossing_point(individual) -> int:
    return np.random.randint(1, len(individual.chromosome))
