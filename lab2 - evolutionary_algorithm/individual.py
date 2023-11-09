from dataclasses import dataclass, field
import numpy as np
from helper_functions import random_number_less_than


@dataclass
class Individual:
    chromosome: list = field(default_factory=list)
    fitness_value: float = field(default=np.inf)

    def calculate_fitness_value(self, func: callable, penalty_func: callable, domain):
        self.fitness_value = func(*self.chromosome) + penalty_func(
            domain, self.chromosome
        )

    def mutate(self, mutation_prob: float, mutation_rate: float):
        for i, gene in enumerate(self.chromosome):
            if random_number_less_than(mutation_prob):
                self.chromosome[i] = gene + mutation_rate * np.random.randn()

    def exchange_genes(self, other, point: int):
        child1_chromosome = self.chromosome[:point] + other.chromosome[point:]
        child2_chromosome = other.chromosome[:point] + self.chromosome[point:]
        child1 = Individual(child1_chromosome)
        child2 = Individual(child2_chromosome)
        return child1, child2
