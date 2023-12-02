from matplotlib import pyplot as plt


def plot(results):
    _, ax = plt.subplots(figsize=(10, 1.7))
    ax.scatter(
        [x.initial_tokens_number for x in results],
        [x.winner.player_id for x in results],
    )
    ax.set_xlabel("Initial number of tokens")
    ax.set_ylabel("Winner")
    ax.set_title("Winner depending on the initial number of tokens")
    plt.yticks([1, 2], ["Player 1", "Player 2"])
    plt.xticks([x.initial_tokens_number for x in results])
    plt.show()
