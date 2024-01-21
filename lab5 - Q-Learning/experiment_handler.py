from dataclasses import dataclass, field
from q_learning import QLearning, calc_avrg_reward_per_100_episodes
import numpy as np


@dataclass
class ExpResult:
    alpha: float
    gamma: float
    avrg_rewards_per_100_episodes: np.ndarray = field(default_factory=np.ndarray)


@dataclass
class ExperimentHandler:
    learning_rates: list = field(default_factory=list)
    discount_factors: list = field(default_factory=list)

    def run_experiments_learning_rates(self):
        results = []
        for learning_rate in self.learning_rates:
            exp = QLearning(alpha=learning_rate)
            _, _, rewards = exp.train()
            avrg_rewards = calc_avrg_reward_per_100_episodes(rewards)
            results.append(ExpResult(learning_rate, exp.gamma, avrg_rewards))
        return results

    def run_experiments_discount_factors(self):
        results = []
        for discount_factor in self.discount_factors:
            exp = QLearning(gamma=discount_factor)
            _, _, rewards = exp.train()
            avrg_rewards = calc_avrg_reward_per_100_episodes(rewards)
            results.append(ExpResult(exp.alpha, discount_factor, avrg_rewards))
        return results
