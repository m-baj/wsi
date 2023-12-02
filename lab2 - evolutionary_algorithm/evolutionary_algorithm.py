from population import Population
from individual import Individual
from iteration import Iteration
import numpy as np
from dataclasses import dataclass, field
from helper_functions import random_number_less_than, get_crossing_point

NUMBER_OF_COMPETITORS = 2


@dataclass
class EvolutionaryAlgorithm:
    fitness_func: callable
    inital_population: Population
    mutation_rate: float = field(default=1)
    mutation_prob: float = field(default=0.8)
    max_iterations: int = field(default=100)
    crossing_prob: float = field(default=0.8)

    def run(self) -> tuple[Individual, list[Iteration]]:
        minimalization_data = []
        curr_iteration = 1
        self.inital_population.calculate_fitness_values(
            self.fitness_func, penalty_function, self.fitness_func.domain
        )
        current_population = self.inital_population
        current_best_individual = self.inital_population.get_fittest()

        while curr_iteration <= self.max_iterations:
            reproduced = self._reproduce(current_population, NUMBER_OF_COMPETITORS)
            crossed = self._cross(reproduced)
            crossed.calculate_fitness_values(
                self.fitness_func, penalty_function, self.fitness_func.domain
            )
            new_potential_best_individual = crossed.get_fittest()
            current_population = crossed

            if self._new_better_individual(
                new_potential_best_individual, current_best_individual
            ):
                current_best_individual = new_potential_best_individual

            iteration = Iteration(
                curr_iteration, current_best_individual, current_population
            )

            minimalization_data.append(iteration)
            curr_iteration += 1

        return current_best_individual, minimalization_data

    def _reproduce(self, population: Population, k: int) -> Population:
        new_population = Population()
        for _ in range(len(population.individuals)):
            competitors = population.choose_random_individuals(k)
            winner = min(competitors, key=lambda i: i.fitness_value)
            new_population.add_individuals(winner)
        return new_population

    def _cross(self, population: Population) -> Population:
        new_population = Population()
        for _ in range(len(population.individuals) // 2):
            individual, partner = population.choose_random_individuals(2)
            if random_number_less_than(self.crossing_prob):
                child1, child2 = self._do_crossing(individual, partner)
                new_population.add_individuals(child1, child2)
            else:
                self._mutate([individual, partner])
                new_population.add_individuals(individual, partner)
        return new_population

    def _do_crossing(
        self, individual: Individual, partner: Individual
    ) -> tuple[Individual, Individual]:
        crossing_point = get_crossing_point(individual)
        child1, child2 = individual.exchange_genes(partner, crossing_point)
        self._mutate([child1, child2])
        return child1, child2

    def _new_better_individual(
        self, new_individual: Individual, current_best_individual: Individual
    ) -> bool:
        return new_individual.fitness_value < current_best_individual.fitness_value

    def _mutate(self, individuals: list[Individual]) -> None:
        for individual in individuals:
            individual.mutate(self.mutation_prob, self.mutation_rate)


def penalty_function(domain, chromosome):
    penalty = 0
    for i, gene in enumerate(chromosome):
        if gene < domain[i][0]:
            penalty += (domain[i][0] - gene) ** 2
        elif gene > domain[i][1]:
            penalty += (gene - domain[i][1]) ** 2
    return penalty
