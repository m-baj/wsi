import pytest
from individual import Individual
import numpy as np


def test_default_individual():
    ind = Individual()
    assert len(ind.chromosome) == 0


def test_individual():
    chromosome = [1, 2, 3, 4]
    ind = Individual(chromosome)
    assert len(ind.chromosome) == 4


def test_default_fitness_val():
    chromosome = [1, 2, 3, 4]
    ind = Individual(chromosome)
    assert ind.fitness_value == np.inf


def test_calc_fitness_val():
    chromosome = [1, 2, 3, 4]
    ind = Individual(chromosome)

    def f(a, b, c, d):
        return a + b + c + d

    ind.calculate_fitness_value(f)
    assert ind.fitness_value == 10


def test_exchange_genes():
    chr1 = [1, 2, 3, 4]
    ind1 = Individual(chr1)
    chr2 = [4, 3, 2, 1]
    ind2 = Individual(chr2)

    child1, child2 = ind1.exchange_genes(ind2, 2)
    assert child1.chromosome == [1, 2, 2, 1]
    assert child2.chromosome == [4, 3, 3, 4]


def test_exchange_genes2():
    chr1 = [1, 3, 3, 4, 2, 7, 5]
    ind1 = Individual(chr1)
    chr2 = [4, 3, 2, 1, 1, 9, 4]
    ind2 = Individual(chr2)

    child1, child2 = ind1.exchange_genes(ind2, 5)
    assert child1.chromosome == [1, 3, 3, 4, 2, 9, 4]
    assert child2.chromosome == [4, 3, 2, 1, 1, 7, 5]


def test_mutation():
    chromosome = [1, 2, 3, 4]
    ind = Individual(chromosome)
    ind.mutate(1, 0.8)
    assert ind.chromosome != [1, 2, 3, 4]
