import warnings
import itertools
from functools import partial

import mne
import numpy as np
import matplotlib.pyplot as plt

from .epoch import make_epochs
from .callback import COLORS, PRINT_KWARGS


def format_arg(arg):
    if isinstance(arg, float):
        return '{:.2f}'.format(arg)
    return arg


def get_label(grid_key, args):
    output = []
    for k in grid_key:
        fmt_string = PRINT_KWARGS.get(k, "{}={{}}".format(k))
        output.append(fmt_string.format(format_arg(args[k])))
    return "\n".join(output)


def _plot_atom(Dk, info, ax, color, plot="atom"):
    sfreq = info.get('sfreq', 1)
    n_channels = info.get('n_channels')
    if plot == "atom":
        t = np.arange(len(Dk) - n_channels) / sfreq
        ax.plot(t, Dk[n_channels:], c=color)
    elif plot == "topo":
        mne.viz.plot_topomap(Dk[:info['n_channels']], info, axes=ax,
                             show=False)
    elif plot == "psd":
        psd = np.abs(np.fft.rfft(Dk[info['n_channels']:])) ** 2
        frequencies = np.linspace(0, sfreq / 2.0, len(psd))
        ax.plot(frequencies, psd, color=color)
        # ax.set(xlabel='Frequencies (Hz)', title='PSD atom final')
        ax.grid(True)
        ax.set_xlim(0, 30)
    else:
        raise NotImplementedError("No plot '{}' for atom".format(plot))


def _plot_activation(z_k, info, ax, color, n_times_atom, t_min=0,
                     plot="density"):
    sfreq = int(info.get('sfreq', 1))
    z_k = z_k.mean(axis=0)  # Average over the epochs
    t = np.arange(len(z_k)) / sfreq + t_min
    if plot == "density":
        blob = np.blackman(n_times_atom)  # bandwidth of n_times_atom
        z_k_smooth = np.convolve(z_k, blob, mode='same')
        ax.fill_between(t, z_k_smooth, color=color, alpha=.5)
    elif plot == "logratio":
        eps = 1e-4
        baseline = z_k[:sfreq]  # Take the first second a a baseline
        mean_baseline = max(baseline.mean(), eps)

        patch = np.ones(sfreq)
        energy = np.maximum(np.convolve(z_k, patch, mode='same'), eps)
        logratio = np.log10(energy / mean_baseline)
        # logratio /= np.std(logratio[:sfreq])
        ax.plot(t[sfreq // 2:-sfreq // 2], logratio[sfreq // 2:-sfreq // 2])
    elif plot == "whiskers":
        # Take 1s at the beginning of the epochs as a baseline
        # Take 1s starting from stim as evoked
        # Take 1s starting 1s after the stim as induced
        baseline = z_k[:sfreq]
        evoked = z_k[(0 < t) * (t < 1)]
        induced = z_k[(1 < t) * (t < 2)]

        ax.boxplot([baseline, evoked, induced], sym='',
                   labels=["baseline", "evoked", "induced"])

        # raise NotImplementedError("Not yet!")
    else:
        raise NotImplementedError("No plot '{}' for activations".format(plot))


def _create_fig(n_cols, n_rows, figname, wrap_col=5, w_plot=5, h_plot=3):
    """Create a figure with subplots

    Parameters
    ----------
    n_cols, n_rows: int, number of rows and column in the plot
    figname: str, name of the figure window
    wrap_col: int, if only 1 row is present, wrap the columns
    w_plot, h_plot: size of the subplots in inches
    """
    wrap_subplots = False
    if n_rows == 1:
        n_rows = n_cols // wrap_col + ((n_cols % wrap_col) != 0)
        n_cols = wrap_col
        wrap_subplots = True

    figsize = (n_cols * w_plot, n_rows * h_plot)

    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, sharey=True,
                             sharex=True, figsize=figsize, num=figname)
    if wrap_subplots:
        axes = [axes.ravel()]
    return fig, axes


def plot_dictionary(data, info, dirname, name='D_hat', plot="atom"):
    if plot == "topo":
        if not isinstance(info, mne.Info):
            warnings.warn("topomaps should only be used when `info` parameter "
                          "is a subclass of `mne.Info`.", UserWarning)
            return
    n_cols = len(data[0][1][name])
    n_rows = len(data)
    figname = "{}_{}".format(name, plot)
    fig, axes = _create_fig(n_cols, n_rows, figname)

    for row_ax, (arg, res) in zip(axes, data):
        # Reset the color_cycle iterator at each iteration
        color_cycle = itertools.cycle(COLORS)
        for color, ax, Dk in zip(color_cycle, row_ax, res[name]):
            if Dk.ndim == 1:
                # ax.plot(t, Dk[info['n_channels']:], c=color)
                _plot_atom(Dk, info, ax, color, plot=plot)
        row_ax[0].set_ylabel(get_label(info['grid_key'], arg),
                             fontsize=24, labelpad=10, rotation="horizontal",
                             horizontalalignment="right",
                             verticalalignment="center")
        row_ax[0].set_yticks([])
    # fig.subplots_adjust(left=.025, bottom=.025)
    plt.pause(.001)
    fig.tight_layout()
    figname = "{}/{}.png".format(dirname, figname)
    fig.savefig(figname, dpi=150)


def plot_activation(data, info, dirname, name, plot="density"):
    n_cols = len(data[0][1][name])  # n_atoms
    n_rows = len(data)
    figname = "{}_{}".format(name, plot)
    fig, axes = _create_fig(n_cols, n_rows, figname)

    t_min = info.get('t_min', 0)
    for row_ax, (arg, res) in zip(axes, data):
        try:
            z = make_epochs(res[name], info, arg['n_times_atom'])
        except Exception:
            z = res[name]
        color_cycle = itertools.cycle(COLORS)
        # z shape (n_trials, n_atoms, n_times_valid)
        for color, ax, z_k in zip(color_cycle, row_ax, z.swapaxes(0, 1)):
            _plot_activation(z_k, info, ax, color,
                             n_times_atom=arg['n_times_atom'], t_min=t_min,
                             plot=plot)
        row_ax[0].set_ylabel(get_label(info['grid_key'], arg),
                             fontsize=24, labelpad=10, rotation="horizontal",
                             horizontalalignment="right",
                             verticalalignment="center")
        row_ax[0].set_yticks([])
    # fig.subplots_adjust(left=.025, bottom=.025)
    plt.pause(.001)
    fig.tight_layout()
    figname = "{}/{}.png".format(dirname, figname)
    fig.savefig(figname, dpi=150)


def plot_convergence_curve(data, info, dirname):
    # plot the convergence curve
    eps = 1e-6

    # compute the best pobj over all methods
    best_pobj = np.min([np.min(r['pobj']) for _, r in data])

    fig = plt.figure("convergence", figsize=(12, 12))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    color_cycle = itertools.cycle(COLORS)
    for (args, res), color in zip(data, color_cycle):
        times = list(np.cumsum(res['times']))
        plt.loglog(
            times, (res['pobj'] - best_pobj) / best_pobj + eps, '.-',
            label=get_label(info['grid_key'], args), color=color,
            linewidth=2)
    plt.xlabel('Time (s)', fontsize=24)
    plt.ylabel('Objective value', fontsize=24)
    ncol = int(np.ceil(len(data) / 10))
    plt.legend(ncol=ncol, fontsize=24)

    plt.gca().tick_params(axis='x', which='both', bottom=False, top=False)
    plt.gca().tick_params(axis='y', which='both', left=False, right=False)
    plt.tight_layout()
    plt.grid(True)
    figname = "{}/convergence.png".format(dirname)
    fig.savefig(figname, dpi=150)


PLOTS = {
    'convergence': plot_convergence_curve,
    'z_init': partial(plot_activation, name='z_init'),
    'z_hat': partial(plot_activation, name='z_hat'),
    'z_hat_whiskers': partial(plot_activation, name='z_hat', plot='whiskers'),
    'z_init_whiskers': partial(plot_activation, name='z_init',
                               plot='whiskers'),
    'z_hat_logratio': partial(plot_activation, name='z_hat', plot='logratio'),
    'z_init_logratio': partial(plot_activation, name='z_init',
                               plot='logratio'),
    'D_init': partial(plot_dictionary, name='D_init'),
    'D_hat': partial(plot_dictionary, name='D_hat'),
    'D_init_psd': partial(plot_dictionary, name='D_init', plot="psd"),
    'D_hat_psd': partial(plot_dictionary, name='D_hat', plot="psd"),
    'D_init_topo': partial(plot_dictionary, name='D_init', plot="topo"),
    'D_hat_topo': partial(plot_dictionary, name='D_hat', plot="topo"),
}


DEFAULT_OUTPUT = [
    'convergence',
    'D_hat_psd', 'D_hat', 'D_init',
    'z_hat', 'z_hat_whiskers'
]
