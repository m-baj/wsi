from dataclasses import dataclass, field
import numpy as np


@dataclass
class State:
    tokens_left: int
    is_max_player_move: bool
    max_tokens_to_take: int
    is_terminal: bool = field(default=False)

    def __post_init__(self):
        self.is_terminal = not self.tokens_left

    def get_child_states(self):
        children_states = []
        for i in range(1, self.max_tokens_to_take + 1):
            child_state = State(
                self.tokens_left - i,
                not self.is_max_player_move,
                min(self.max_tokens_to_take, self.tokens_left - i),
            )
            children_states.append(child_state)
        return children_states


def calculate_heuristic_for(current_state):
    N = current_state.tokens_left
    K = current_state.max_tokens_to_take
    i = (N - K - 2) / (K + 1)
    if i.is_integer():
        return -1
    else:
        return 1


def calculate_value_for(current_state):
    if current_state.is_terminal:
        value = 1 if current_state.is_max_player_move else -1
        return value
    else:
        return calculate_heuristic_for(current_state)


def minimax(current_state: State, depth: int, move_max: bool, alpha: int, beta: int):
    if current_state.is_terminal or not depth:
        return calculate_value_for(current_state)

    child_states = current_state.get_child_states()

    if move_max:
        for state in child_states:
            alpha = max(alpha, minimax(state, depth - 1, not move_max, alpha, beta))
            if alpha >= beta:
                return alpha
        return alpha

    else:
        for state in child_states:
            beta = min(beta, minimax(state, depth - 1, not move_max, alpha, beta))
            if alpha >= beta:
                return beta
        return beta
