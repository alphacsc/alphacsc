import itertools
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

from .viz import plot_activations_density, COLORS


def plot_dictionary(data, info, dirname, name):
    n_cols = len(data[0][1])
    n_rows = len(data)
    sfreq = info.get('sfreq', 1)
    figsize = (n_cols * 5, n_rows * 3)
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, sharey=True,
                             sharex=True, figsize=figsize, num=name)

    for row_ax, (arg, D) in zip(axes, data):
        t = np.arange(arg['n_times_atom']) / sfreq
        # Reset the color_cycle iterator at each iteration
        color_cycle = itertools.cycle(COLORS)
        for i, (color, ax, Dk) in enumerate(zip(color_cycle, row_ax, D)):
            if Dk.ndim == 1:
                ax.plot(t, Dk[info['n_channels']:], c=color)
            if i == 0:
                ax.set_ylabel("$\lambda$: {:.2f}".format(arg['reg']),
                              fontsize=24, labelpad=10, rotation="horizontal",
                              horizontalalignment="right",
                              verticalalignment="center")
                ax.set_yticks([])
    # fig.subplots_adjust(left=.025, bottom=.025)
    plt.pause(.001)
    fig.tight_layout()
    figname = "{}/{}.png".format(dirname, name)
    fig.savefig(figname, dpi=150)


def plot_activation(data, info, dirname, name):
    n_cols = len(data[0][1])
    n_rows = len(data)
    sfreq = info.get('sfreq', 1)
    figsize = (n_cols * 5, n_rows * 3)
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, sharey=True,
                             sharex=True, figsize=figsize, num=name)
    for row_ax, (arg, Z) in zip(axes, data):

        plot_activations_density(Z, n_times_atom=arg['n_times_atom'],
                                 sfreq=sfreq, axes=row_ax)
        row_ax[0].set_ylabel("$\lambda$: {:.2f}".format(arg['reg']),
                             fontsize=24, labelpad=10, rotation="horizontal",
                             horizontalalignment="right",
                              verticalalignment="center")
        row_ax[0].set_yticks([])
    # fig.subplots_adjust(left=.025, bottom=.025)
    plt.pause(.001)
    fig.tight_layout()
    figname = "{}/{}.png".format(dirname, name)
    fig.savefig(figname, dpi=150)


PLOTS = {
    'D_init': partial(plot_dictionary, name='D_init'),
    'D_hat': partial(plot_dictionary, name='D_hat'),
    'Z_init': partial(plot_activation, name='Z_init'),
    'Z_hat': partial(plot_activation, name='Z_hat'),
}


DEFAULT_OUTPUT = ['D_init', 'D_hat', 'Z_init', 'Z_hat']