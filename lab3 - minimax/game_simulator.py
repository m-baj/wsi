from game import Game
from dataclasses import dataclass


@dataclass
class GameSimulator:
    number_of_games: int
    max_number_of_tokens_to_take: int
    ai_depth: int

    def simulate(self):
        game_raports = []
        for _ in range(self.number_of_games):
            game = Game(
                self.max_number_of_tokens_to_take, ai_depth=self.ai_depth, ai_vs_ai=True
            )
            game_raports.append(game.start())
        return game_raports
