from game import Game
from gameUI import UI


def main():
    ui = UI(show_messages=True)
    game = Game(3, ui=ui, ai_depth=100)
    game.start()


if __name__ == "__main__":
    main()
