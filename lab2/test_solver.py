from solver import Solver
import pytest


def test_get_number_of_experiments():
    solver = Solver()
    solver.hyperparameters = {
        "mutation_rate": [0.1, 0.5, 1, 2, 5, 10],
        "mutation_probability": 0.8,
        "max_iterations": 10000,
        "crossing_probability": 0.8,
    }
    assert solver._get_number_of_experiments() == 6


def test_get_number_of_experiments2():
    solver = Solver()
    solver.hyperparameters = {
        "mutation_rate": 0.1,
        "mutation_probability": 0.8,
        "max_iterations": 10000,
        "crossing_probability": 0.8,
    }
    assert solver._get_number_of_experiments() == 1


def test_get_number_of_experiments3():
    solver = Solver()
    solver.hyperparameters = {
        "mutation_rate": 0.1,
        "mutation_probability": [0.8, 0.9],
        "max_iterations": 10000,
        "crossing_probability": 0.8,
    }
    assert solver._get_number_of_experiments() == 2


def test_get_variable_params():
    solver = Solver()
    solver.hyperparameters = {
        "mutation_rate": [0.1, 0.5, 1, 2, 5, 10],
        "mutation_probability": 0.8,
        "max_iterations": 10000,
        "crossing_probability": 0.8,
    }
    var_params = solver._get_variable_params()
    assert var_params == {
        "mutation_rate": [0.1, 0.5, 1, 2, 5, 10],
    }


def test_init_experiments():
    solver = Solver()
    solver.hyperparameters = {
        "mutation_rate": [0.1, 0.5, 1, 2, 5, 10],
        "mutation_probability": 0.8,
        "max_iterations": 10000,
        "crossing_probability": 0.8,
    }
    exps = solver.init_experiments()
    assert len(exps) == 6


def test_init_experiments2():
    solver = Solver()
    solver.hyperparameters = {
        "mutation_rate": 5,
        "mutation_probability": 0.8,
        "max_iterations": 10000,
        "crossing_probability": 0.8,
    }
    exps = solver.init_experiments()
    assert len(exps) == 1
