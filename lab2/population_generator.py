from dataclasses import dataclass, field
from function import Function
from population import Population
from individual import Individual
import numpy as np


@dataclass
class PopulationGenerator:
    function: Function

    def generate_population(self, size: int = 100) -> Population:
        initial_population = Population()
        for _ in range(size):
            chromosome = self._generate_chromosome()
            individual = Individual(chromosome)
            initial_population.add_individuals(individual)
        return initial_population

    def _generate_chromosome(self) -> list:
        chromosome = []
        for i in range(self.function.number_of_variables):
            chromosome.append(self._generate_gene(i))
        return chromosome

    def _generate_gene(self, number_of_variable: int) -> float:
        domain = self.function.domain[number_of_variable]
        lower_bound = domain[0]
        upper_bound = domain[1]
        gene = np.random.uniform(lower_bound, upper_bound)
        return gene


@dataclass
class SpecialPopulationGenerator:
    def generate_population(self, point, size: int = 100) -> Population:
        initial_population = Population()
        for _ in range(size):
            first_gene = np.random.normal(point[0], 1)
            second_gene = np.random.normal(point[1], 1)
            chromosome = [
                first_gene,
                second_gene,
            ]
            individual = Individual(chromosome)
            initial_population.add_individuals(individual)
        return initial_population
