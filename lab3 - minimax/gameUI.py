from dataclasses import dataclass, field


@dataclass
class UI:
    show_messages: bool = field(default=False)

    def _log_wrapper(function):
        def wrapper(self, *args, **kwargs):
            if self.show_messages:
                return function(self, *args, **kwargs)

        return wrapper

    @_log_wrapper
    def print_tokens_number(self, current_tokens_number):
        print(f"\tCurrent tokens number: {current_tokens_number}")

    @_log_wrapper
    def print_tokens_taken(self, player, tokens_taken, tokens_left):
        if not tokens_left:
            print(f"\tPlayer {player} took last token\n")
        s = "" if tokens_taken == 1 else "s"
        print(f"\tPlayer {player} took {tokens_taken} token{s}\n")

    @_log_wrapper
    def print_max_tokens_to_remove(self, max_tokens_to_remove):
        print(f"Maximum number of tokens to take: {max_tokens_to_remove}\n")

    @_log_wrapper
    def print_round_number(self, round_number):
        print(f"Round {round_number}:")

    @_log_wrapper
    def print_loser(self, player):
        print(f"Player {player} lost!\n")

    @_log_wrapper
    def print_winner(self, player):
        print(f"The winner is player {player}!")

    @_log_wrapper
    def print_message(self, message):
        print(message)

    def get_tokens_to_remove(self):
        while True:
            try:
                tokens_number = int(input(f"Input amount of tokens to take: "))
                return tokens_number
            except ValueError:
                print("Invalid input\n")
                continue
