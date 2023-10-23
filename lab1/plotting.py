import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

FIGURE_SIZE = (10, 6)


def plot_2d(x, function, domain, label="", title="", path_to_min=None):
    fig = plt.figure(figsize=FIGURE_SIZE)
    fig.suptitle(title, fontsize="xx-large")
    ax = fig.add_subplot(111)
    ax.plot(x, function(x), label=label)
    left, right = domain
    plt.xlim(left, right)
    ax.set_xlabel("x", fontsize="x-large")
    ax.set_ylabel("y", fontsize="x-large", rotation=0)
    ax.legend(loc="upper center", fontsize="x-large")

    ax.grid()

    if path_to_min is not None:
        ax.plot(
            path_to_min,
            function(path_to_min),
            marker="o",
            color="red",
            label="path to min",
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

    ax.view_init(40, 40)
    ax.set_zlim(0, 10)

    ax.grid()

    ax2 = fig.add_subplot(122)
    ax2.contour(x, y, function(x, y), cmap="viridis", levels=20)

    if path_to_min is not None:
        print(path_to_min[:, 0])
        print(path_to_min[:, 1])
        for ax in [ax, ax2]:
            ax.plot(
                path_to_min[:, 0],
                path_to_min[:, 1],
                function(path_to_min[:, 0], path_to_min[:, 1]),
                marker="o",
                color="red",
                label="path to min",
            )

    plt.show()
