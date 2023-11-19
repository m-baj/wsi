from dataclasses import dataclass, field


@dataclass
class GameResult:
    initial_tokens_number: int
    max_tokens_to_take: int
    winner: int = field(default=None)
