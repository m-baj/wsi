from dataclasses import dataclass, field
from player import Player, MiniMaxPlayer, InvalidInput
from result import GameResult
import numpy as np
from gameUI import UI


NUMBER_OF_TOKENS_LOWER = 5
NUMBER_OF_TOKENS_UPPER = 20


@dataclass
class Game:
    max_number_of_tokens_to_take: int
    tokens_left: int = field(default=None)
    players: list[Player] = field(default_factory=list)
    ai_vs_ai: bool = field(default=False)
    ai_depth: int = field(default=100)
    ui: UI = field(default=UI())

    def __post_init__(self):
        self._create_players()
        if self.tokens_left is None:
            self.tokens_left = generate_tokens_number(
                NUMBER_OF_TOKENS_LOWER, NUMBER_OF_TOKENS_UPPER
            )

    def start(self):
        game_raport = GameResult(self.tokens_left, self.max_number_of_tokens_to_take)
        self.ui.print_max_tokens_to_remove(self.max_number_of_tokens_to_take)
        round_number = 1
        current_player = self.players[1]
        while True:
            self.ui.print_round_number(round_number)
            self.ui.print_tokens_number(self.tokens_left)

            current_player = self._get_next_player(current_player)
            tokens_to_remove = self._ask_for_tokens_and_check(current_player)
            self.tokens_left -= tokens_to_remove
            self.ui.print_tokens_taken(
                current_player, tokens_to_remove, self.tokens_left
            )

            if self.tokens_left == 0:
                game_raport.winner = self._get_winner(current_player)
                self.ui.print_loser(current_player)
                self.ui.print_winner(self._get_winner(current_player))
                return game_raport

            round_number += 1

    def _ask_for_tokens_and_check(self, player):
        while True:
            try:
                tokens_to_remove = self._ask_for_tokens(player)
                break
            except InvalidInput as e:
                self.ui.print_message(e)
                continue
        return tokens_to_remove

    def _ask_for_tokens(self, player):
        return player.take_tokens(self)

    def _create_players(self):
        if self.ai_vs_ai:
            self.players = [
                MiniMaxPlayer(1, self.ai_depth),
                MiniMaxPlayer(2, self.ai_depth),
            ]
        else:
            self.players = [Player(1), MiniMaxPlayer(2, self.ai_depth)]

    def _get_next_player(self, current_player):
        if current_player == self.players[0]:
            return self.players[1]
        else:
            return self.players[0]

    def _get_winner(self, current_player):
        if not self.tokens_left:
            return self._get_next_player(current_player)


def generate_tokens_number(lower, upper) -> int:
    return np.random.randint(lower, upper + 1)
