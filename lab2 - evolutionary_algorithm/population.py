from dataclasses import dataclass, field
import numpy as np
from individual import Individual


@dataclass
class Population:
    individuals: list[Individual] = field(default_factory=list)
    best_individual: Individual = field(default=None)

    def get_fittest(self) -> Individual:
        return min(self.individuals, key=lambda i: i.fitness_value)

    def calculate_fitness_values(self, func: callable, penalty_func: callable, domain):
        for individual in self.individuals:
            individual.calculate_fitness_value(func, penalty_func, domain)

    def choose_random_individuals(self, number: int = 1) -> list[Individual]:
        return np.random.choice(self.individuals, number)

    def add_individuals(self, *new_individuals: list[Individual]) -> None:
        for individual in new_individuals:
            self.individuals.append(individual)

    def average_fitness(self) -> float:
        return np.mean([i.fitness_value for i in self.individuals])
