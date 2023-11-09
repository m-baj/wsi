from evolutionary_algorithm import penalty_function
import pytest


def test_penalty_function():
    chromosome = [1, 2, 3, 4]
    domain = [[0, 5], [0, 5], [0, 5], [0, 5]]
    assert penalty_function(domain, chromosome) == 0
    chromosome = [-1, 2, 3, 4]
    result = penalty_function(domain, chromosome)
    assert result == 1


def test_penalty_function_raises_exception():
    chromosome = [100, 2, 3, 4]
    domain = [[0, 5], [0, 5], [0, 5], [0, 5]]
