import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

FIGURE_SIZE = (8, 6)


def plot_2d(x, function, label="", title="", path_to_min=None):
    fig = plt.figure(figsize=FIGURE_SIZE)
    fig.suptitle(title, fontsize="xx-large")
    ax = fig.add_subplot(111)
    ax.plot(x, function(x), label=label)
    plt.xlim(-3, 3)
    ax.set_xlabel("x", fontsize="x-large")
    ax.set_ylabel("y", fontsize="x-large", rotation=0)
    ax.legend(loc="upper center", fontsize="xx-large")

    ax.grid()

    if path_to_min is not None:
        ax.plot(
            path_to_min,
            function(path_to_min),
            marker="o",
            color="red",
            label="path",
        )

    plt.show()


def plot_3d(x, y, function, title="", path_to_min=None):
    fig = plt.figure(figsize=FIGURE_SIZE)
    fig.suptitle(title, fontsize="xx-large")
    ax = fig.add_subplot(121, projection="3d")

    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, function(x, y), cmap="viridis", edgecolor="none")

    ax.set_xlabel("x", fontsize="x-large")
    ax.set_ylabel("y", fontsize="x-large")
    ax.set_zlabel("z", fontsize="x-large", rotation=0)

    ax.set_zlim(0, 200)

    ax.grid()

    ax2 = fig.add_subplot(122)
    ax2.contour(
        x,
        y,
        function(x, y),
        cmap="viridis",
        levels=20,
    )
    ax2.set_xlabel("x", fontsize="x-large")
    ax2.set_ylabel("y", fontsize="x-large", rotation=0, labelpad=0)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    if path_to_min is not None:
        ax.plot(
            path_to_min[:, 0],
            path_to_min[:, 1],
            function(path_to_min[:, 0], path_to_min[:, 1]),
            marker="o",
            color="red",
            label="path to min",
        )

        ax2.plot(
            path_to_min[:, 0],
            path_to_min[:, 1],
            marker="o",
            color="red",
            label="path to min",
        )

    plt.show()
