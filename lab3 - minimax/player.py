from dataclasses import dataclass
import numpy as np
from minimax import minimax, State

DEPTH = 100
INITIAL_ALPHA = -np.inf
INITIAL_BETA = np.inf


class InvalidInput(Exception):
    pass


@dataclass
class Player:
    player_id: int

    def take_tokens(self, game):
        tokens_number = game.ui.get_tokens_to_remove()
        max_tokens_to_remove = min(game.tokens_left, game.max_number_of_tokens_to_take)
        if tokens_number > max_tokens_to_remove:
            raise InvalidInput(
                f"\nInvalid input. Maximum number of tokens you can take: {max_tokens_to_remove}\n"
            )
        elif tokens_number < 1:
            raise InvalidInput("\nInvalid input. You must take at least 1 token\n")
        return tokens_number

    def __str__(self):
        return str(self.player_id)


@dataclass
class MiniMaxPlayer(Player):
    depth: int

    def take_tokens(self, game):
        max_tokens_to_remove = min(game.tokens_left, game.max_number_of_tokens_to_take)
        tokens_number = self._calculate_tokens_to_remove(
            game.tokens_left, max_tokens_to_remove
        )
        return tokens_number

    def _calculate_tokens_to_remove(self, tokens_left, max_tokens_to_remove):
        current_state = State(tokens_left, True, max_tokens_to_remove)
        best_value = INITIAL_ALPHA
        best_move = None
        for state in current_state.get_child_states():
            value = minimax(state, self.depth, False, INITIAL_ALPHA, INITIAL_BETA)
            if value > best_value:
                best_value = value
                best_move = state.tokens_left
        return tokens_left - best_move
