from evolutionary_algorithm import EvolutionaryAlgorithm
from population import Population
from individual import Individual
from function import Function
from dataclasses import dataclass, field
from experiment_result import ExpResult

MUTATION_RATES = [0.1, 0.5, 1, 2, 5, 10]

HYPERPARAMETERS = {
    "mutation_rate": MUTATION_RATES,
    "mutation_probability": 0.8,
    "max_iterations": 10000,
    "crossing_probability": 0.8,
}


@dataclass
class Solver:
    hyperparameters: dict = field(default_factory=dict)
    function: Function = field(default=None)
    population: Population = field(default=None)

    def run_experiments(
        self, experiments: list[EvolutionaryAlgorithm]
    ) -> list[ExpResult]:
        experiments_results = []
        for i, exp in enumerate(experiments):
            result = ExpResult(i, exp, *exp.run())
            experiments_results.append(result)
        return experiments_results

    def run_experiment_n_times(self, experiment, n: int) -> list[ExpResult]:
        experiments_resutls = []
        for i in range(n):
            result = ExpResult(i, experiment, *experiment.run())
            experiments_resutls.append(result)
        return experiments_resutls

    def init_experiments(self) -> list[EvolutionaryAlgorithm]:
        experiments = []
        variable_hyperparameters = self._get_variable_params()
        number_of_experiments = self._get_number_of_experiments()
        for _ in range(number_of_experiments):
            if "mutation_rate" in variable_hyperparameters:
                mutation_rate = next(variable_hyperparameters["mutation_rate"])
            else:
                mutation_rate = self.hyperparameters["mutation_rate"]

            if "mutation_probability" in variable_hyperparameters:
                mutation_probability = next(
                    variable_hyperparameters["mutation_probability"]
                )
            else:
                mutation_probability = self.hyperparameters["mutation_probability"]

            if "max_iterations" in variable_hyperparameters:
                max_iterations = next(variable_hyperparameters["max_iterations"])
            else:
                max_iterations = self.hyperparameters["max_iterations"]

            if "crossing_probability" in variable_hyperparameters:
                crossing_probability = next(
                    variable_hyperparameters["crossing_probability"]
                )
            else:
                crossing_probability = self.hyperparameters["crossing_probability"]

            exp = EvolutionaryAlgorithm(
                self.function,
                self.population,
                mutation_rate,
                mutation_probability,
                max_iterations,
                crossing_probability,
            )
            experiments.append(exp)
        return experiments

    def _get_variable_params(self) -> dict[str, iter]:
        return {
            param: iter(self.hyperparameters[param])
            for param in self.hyperparameters
            if isinstance(self.hyperparameters[param], list)
        }

    def _get_number_of_experiments(self) -> int:
        if not self._check_if_any_variable_params():
            return 1
        return max(
            [
                len(param)
                for param in self.hyperparameters.values()
                if isinstance(param, list)
            ]
        )

    def _check_if_any_variable_params(self) -> bool:
        return len(self._get_variable_params()) > 0
