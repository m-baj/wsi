import pytest
from population import Population
from individual import Individual


def test_population():
    pop = Population()
    assert pop.individuals == []


def test_add_individual():
    pop = Population()
    ind1 = Individual()
    ind2 = Individual()
    ind3 = Individual()
    pop.add_individuals(ind1, ind2, ind3)
    assert pop.individuals == [ind1, ind2, ind3]
    assert len(pop.individuals) == 3


def test_choose_random_individual_except():
    pop = Population()
    ind1 = Individual([1, 2])
    ind2 = Individual([2, 3])
    ind3 = Individual([4, 5])
    pop.add_individuals(ind1, ind2, ind3)
    ind = pop.choose_random_individual_except([ind2])
    assert ind in [ind1, ind3]
