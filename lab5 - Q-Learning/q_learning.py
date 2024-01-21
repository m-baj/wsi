import gymnasium as gym
import numpy as np
from dataclasses import dataclass, field

MIN_EXPLORATION_RATE = 0.01
MAX_EXPLORATION_RATE = 1.0
EXPLORATION_DECAY_RATE = 0.001
MAX_STEPS_PER_EPISODE = 100


@dataclass
class QLearning:
    alpha: float = 0.1
    gamma: float = 0.6
    epsilon: float = 1.0
    env: gym.Env = field(default_factory=lambda: gym.make("CliffWalking-v0"))

    def __post_init__(self):
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def train(self, episodes=10000):
        epsilons = []
        rewards = []
        for episode in range(episodes):
            state = self.env.reset()[0]
            reward = 0
            for step in range(MAX_STEPS_PER_EPISODE):
                action = self._choose_action(state)
                next_state, curr_reward, done, _, info = self.env.step(action)

                self._update_q_table(state, action, curr_reward, next_state)
                reward += curr_reward

                state = next_state
                if done:
                    break
            epsilons.append(self.epsilon)
            rewards.append(reward)
            self._decay_exploration_rate(episode)
        return self.q_table, epsilons, rewards

    def _choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state, :])
        return action

    def _update_q_table(self, state, action, reward, next_state):
        self.q_table[state, action] += self.alpha * (
            reward
            + self.gamma * np.max(self.q_table[next_state, :])
            - self.q_table[state, action]
        )

    def _decay_exploration_rate(self, episode):
        self.epsilon = MIN_EXPLORATION_RATE + (
            MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE
        ) * np.exp(-EXPLORATION_DECAY_RATE * episode)


def play(env, q_table):
    state = env.reset()[0]
    done = False
    while not done:
        action = np.argmax(q_table[state, :])
        next_state, reward, done, _, info = env.step(action)
        state = next_state
        env.render()
    env.close()


def calc_avrg_reward_per_100_episodes(rewards):
    avrg_reward_per_100_episodes = np.array_split(np.array(rewards), len(rewards) / 100)
    return [np.mean(x) for x in avrg_reward_per_100_episodes]
