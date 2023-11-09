from dataclasses import dataclass, field
from individual import Individual
from evolutionary_algorithm import EvolutionaryAlgorithm
from iteration import Iteration


@dataclass
class ExpResult:
    exp_id: int
    experiment: EvolutionaryAlgorithm
    best_individual: Individual
    data: list[Iteration] = field(default_factory=list)
