from dataclasses import dataclass, field
from individual import Individual


@dataclass
class Iteration:
    iteration_number: int
    best_individual: Individual
    population: list[Individual] = field(default_factory=list)
