from population_generator import PopulationGenerator
from function import Function
import pytest


def test_population_generator():
    func = Function(lambda x: x**2, [(-10, 10)], 1)
    pop_gen = PopulationGenerator(func)
    assert pop_gen.population_size == 100


def test_population_generator2():
    func = Function(lambda x: x**2, [(-10, 10)], 1)
    pop_gen = PopulationGenerator(func, 10)
    pop = pop_gen.generate()
    assert len(pop.individuals) == 10


def test_population_generator3():
    func = Function(lambda x, y: x**2 + y**2, [(-6, -5), (5, 6)], 2)
    pop_gen = PopulationGenerator(func, 10)
    pop = pop_gen.generate()
    assert len(pop.individuals) == 10


def test_population_generator4():
    func = Function(
        lambda x, y, z: x**2 + y**2 + z**2, [(1, 2), (3, 4), (5, 6)], 3
    )
    pop_gen = PopulationGenerator(func, 100000)
    pop = pop_gen.generate()
    ind = pop.choose_random_individuals()[0]
    assert ind.chromosome[0] >= 1 and ind.chromosome[0] <= 2
