import warnings
import itertools
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

from .signal import make_epochs
from .viz import plot_activations_density, COLORS, PRINT_KWARGS


def _check_mne_and_info(info):
    """Check that mne is installed and the the info parameter is a `mne.Info`
    """
    try:
        import mne
    except ImportError:
        warnings.warn("mne is required to visualize the topomap of the "
                      "atoms", UserWarning)
        return False
    if not isinstance(info, mne.Info):
        warnings.warn("topomaps should only be used when `info` parameter "
                      "is a subclass of `mne.Info`.", UserWarning)
        return False
    return True


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
        # We can import mne here as we already checked that mne was installed
        from mne.viz import plot_topomap
        plot_topomap(Dk[:info['n_channels']], info, axes=ax, show=False)
    elif plot == "psd":
        psd = np.abs(np.fft.rfft(Dk[info['n_channels']:])) ** 2
        frequencies = np.linspace(0, sfreq / 2.0, len(psd))
        ax.plot(frequencies, psd, color=color)
        # ax.set(xlabel='Frequencies (Hz)', title='PSD atom final')
        ax.grid(True)
        ax.set_xlim(0, 30)
    else:
        raise NotImplementedError("No plot '{}' for atom".format(plot))


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
    if plot == "topo" and not _check_mne_and_info(info):
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


def plot_activation(data, info, dirname, name):
    n_cols = len(data[0][1][name])  # n_atoms
    n_rows = len(data)
    fig, axes = _create_fig(n_cols, n_rows, name)

    sfreq = info.get('sfreq', 1)
    for row_ax, (arg, res) in zip(axes, data):
        try:
            Z = make_epochs(res[name], info, arg['n_times_atom'])
        except BaseException:
            Z = res[name]
        t_min = info.get('t_min', 0)
        plot_activations_density(Z, n_times_atom=arg['n_times_atom'],
                                 sfreq=sfreq, axes=row_ax, t_min=t_min)
        row_ax[0].set_ylabel(get_label(info['grid_key'], arg),
                             fontsize=24, labelpad=10, rotation="horizontal",
                             horizontalalignment="right",
                             verticalalignment="center")
        row_ax[0].set_yticks([])
    # fig.subplots_adjust(left=.025, bottom=.025)
    plt.pause(.001)
    fig.tight_layout()
    figname = "{}/{}.png".format(dirname, name)
    fig.savefig(figname, dpi=150)


def plot_convergence_curve(data, info, dirname):
    # plot the convergence curve

    # compute the best pobj over all methods
    best_pobj = np.min([np.min(r['pobj']) for _, r in data]) - 1e-6

    fig = plt.figure("convergence", figsize=(12, 12))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    color_cycle = itertools.cycle(COLORS)
    for (args, res), color in zip(data, color_cycle):
        times = list(np.cumsum(res['times']))
        plt.semilogx(
            times, res['pobj'] - best_pobj, '.-', alpha=0.5,
            label=get_label(info['grid_key'], args), color=color)
    plt.xlabel('Time (s)')
    plt.ylabel('Objective value')
    ncol = int(np.ceil(len(data) / 10))
    plt.legend(ncol=ncol)

    plt.gca().tick_params(axis='x', which='both', bottom=False, top=False)
    plt.gca().tick_params(axis='y', which='both', left=False, right=False)
    plt.tight_layout()
    plt.grid(True)
    figname = "{}/convergence.png".format(dirname)
    fig.savefig(figname, dpi=150)


PLOTS = {
    'convergence': plot_convergence_curve,
    'D_init': partial(plot_dictionary, name='D_init'),
    'D_hat': partial(plot_dictionary, name='D_hat'),
    'D_init_psd': partial(plot_dictionary, name='D_init', plot="psd"),
    'D_hat_psd': partial(plot_dictionary, name='D_hat', plot="psd"),
    'D_init_topo': partial(plot_dictionary, name='D_init', plot="topo"),
    'D_hat_topo': partial(plot_dictionary, name='D_hat', plot="topo"),
    'Z_init': partial(plot_activation, name='Z_init'),
    'Z_hat': partial(plot_activation, name='Z_hat'),
}


DEFAULT_OUTPUT = [
    'convergence',
    'D_init_topo', 'D_hat_topo', 'D_init_psd', 'D_hat_psd', 'D_hat', 'D_init',
    'Z_init', 'Z_hat'
]