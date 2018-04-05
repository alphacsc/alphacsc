import itertools

import copy as cp

import os
import mne
import json
import numpy as np
import matplotlib as mpl
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

from .tools import get_calling_script, positive_hash

COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

DEFAULT_CB = {
    'atom': {},
    'Zhat': {},
    'pobj': {}
}
PRINT_KWARGS = {
    'n_atoms': 'K = {}',
    'n_times_atom': 'L={}',
    'reg': '$\lambda$={}',
    'solver_z': '{}'
}


mpl.rc('mathtext', fontset='cm')


def kde_sklearn(x, x_grid, bandwidth):
    """Kernel Density Estimation with Scikit-learn"""
    n_samples, = x.shape
    if n_samples == 0:
        return np.zeros_like(x_grid)

    window = np.blackman(bandwidth) * bandwidth
    return np.convolve(x, window, 'same')


def plot_activations_density(Z_hat, n_times_atom, sfreq=1., threshold=0.01,
                             bandwidth='auto', axes=None, t_min=0,
                             plot_activations=False, colors=None):
    """
    Parameters
    ----------
    Z_hat : array, shape (n_atoms, n_trials, n_times_valid)
        The sparse activation matrix.
    n_times_atom : int
        The support of the atom.
    sfreq : float
        Sampling frequency
    threshold : float
        Remove activations (normalized with the max) below this threshold
    bandwidth : float, array of float, or 'auto'
        Bandwidth (in sec) of the kernel
    axes : array of axes, or None
        Axes to plot into
    t_min : float
        Time offset for the xlabel display
    plot_activations : boolean
        If True, the significant activations are plotted as black dots
    colors : list of matplotlib compatible colors
        Colors of the plots
    """
    n_atoms, n_trials, n_times_valid = Z_hat.shape

    # sum activations over all trials
    Z_hat_sum = Z_hat.sum(axis=1)

    if bandwidth == 'auto':
        bandwidth = n_times_atom

    if axes is None:
        fig, axes = plt.subplots(n_atoms, num='density',
                                 figsize=(8, 2 + n_atoms * 3))
    axes = np.atleast_1d(axes)

    if colors is None:
        colors = itertools.cycle(COLORS)
    for ax, activations, color in zip(axes.ravel(), Z_hat_sum, colors):
        ax.clear()
        time_instants = np.arange(n_times_valid) / float(sfreq) + t_min
        selection = activations > threshold * Z_hat_sum.max()
        n_elements = selection.sum()

        if n_elements == 0:
            ax.plot(time_instants, np.zeros_like(time_instants))
            continue

        # plot the activations as black dots
        if plot_activations:
            ax.plot(time_instants[selection],
                    activations[selection] / activations[selection].max(), '.',
                    color='k')

        window = np.blackman(bandwidth)
        smooth_activations = np.convolve(activations, window, 'same')
        ax.fill_between(time_instants, smooth_activations, color=color,
                        alpha=0.5)

    return axes


def try_plot_activations_density():
    """Just try it"""
    Z_hat = np.random.randn(3, 1, 1000)
    Z_hat[Z_hat < 2] = 0
    Z_hat[:, :, :500] /= 6.

    plot_activations_density(Z_hat, 150, plot_activations=True)
    plt.show()


def plot_data(X, plot_types=None):
    """Plot the data.

    Parameters
    ----------
    X : list
        A list of arrays of shape (n_trials, n_times).
        E.g., one could give [X, X_hat]
    plot_types : list of str
        If None, plt.plot for all.
        E.g., plot_data([X, Z], ['plot', 'stem'])
    """
    import matplotlib.pyplot as plt

    if not isinstance(X, list):
        raise ValueError('Got %s. It must be a list' % type(X))

    if plot_types is None:
        plot_types = ['plot' for ii in range(len(X))]

    if not isinstance(plot_types, list):
        raise ValueError('Got %s. It must be a list' % type(plot_types))
    if len(plot_types) != len(X):
        raise ValueError('X and plot_types must be of same length')

    def _onclick(event):
        orig_ax = event.inaxes
        fig, ax = plt.subplots(1)
        ax.set_axis_bgcolor('white')
        for jj in range(len(plot_types)):
            if orig_ax._plot_types[jj] == 'plot':
                ax.plot(orig_ax._X[jj])
            elif orig_ax._plot_types[jj] == 'stem':
                ax.plot(orig_ax._X[jj], '-o')
        plt.title('%s' % orig_ax._name)
        plt.show()

    n_trials = X[0].shape[0]
    fig, axes = plt.subplots(n_trials, 1, sharex=True, sharey=True)
    fig.canvas.mpl_connect('button_press_event', _onclick)
    fig.patch.set_facecolor('white')
    for ii in range(n_trials):
        for jj in range(len(X)):
            if plot_types[jj] == 'plot':
                axes[ii].plot(X[jj][ii])
            elif plot_types[jj] == 'stem':
                axes[ii].plot(X[jj][ii], '-o')
        axes[ii].get_yaxis().set_ticks([])
        axes[ii].set_ylabel('Trial %d' % (ii + 1), rotation=0, ha='right')
        axes[ii]._name = 'Trial %d' % (ii + 1)
        axes[ii]._plot_types = plot_types
        axes[ii]._X = [X[jj][ii] for jj in range(len(X))]
    plt.xlabel('Time')
    plt.show()


def plot_callback(X, info, n_atoms, layout=None):
    n_trials, n_chan, n_times = X.shape

    n_atoms_plot = min(15, n_atoms)

    fig, axes = plt.subplots(nrows=n_atoms, num='atoms', figsize=(10, 8))
    fig_Z, axes_Z = plt.subplots(nrows=n_atoms, num='Z', figsize=(10, 8),
                                 sharex=True, sharey=True)
    fig_topo, axes_topo = plt.subplots(1, n_atoms_plot, figsize=(12, 3))

    if n_atoms == 1:
        axes_topo, axes = [axes_topo], [axes]
        if n_trials == 1:
            axes_Z = [axes_Z]

    if layout is None:
        layout = mne.channels.find_layout(info)

    def callback(X, uv_hat, Z_hat, reg):
        n_times_valid = Z_hat.shape[-1]
        n_times_atom = uv_hat.shape[1] - n_chan
        times_Z = np.arange(n_times_valid) / info['sfreq']
        times_v = np.arange(n_times_atom) / info['sfreq']

        this_info = cp.deepcopy(info)
        this_info['sfreq'] = 1.
        patterns = mne.EvokedArray(uv_hat[:n_atoms_plot, :n_chan].T,
                                   this_info, tmin=0)
        patterns.plot_topomap(times=np.arange(n_atoms_plot),
                              layout=layout, axes=axes_topo, scaling_time=1,
                              time_format='Atom%01d', show=False)

        if axes[0].lines == []:
            for k in range(n_atoms):
                axes[k].plot(times_v, uv_hat[k, n_chan:].T)
                axes[k].grid(True)
        else:
            for ax, uv in zip(axes, uv_hat):
                ax.lines[0].set_ydata(uv[n_chan:])
                ax.relim()  # make sure all the data fits
                ax.autoscale_view(True, True, True)
        if n_trials == 1:
            if axes_Z[0].lines == []:
                for k in range(n_atoms):
                    axes_Z[k].plot(times_Z, Z_hat[k, 0])
                    axes_Z[k].grid(True)
            else:
                for ax, z in zip(axes_Z, Z_hat[:, 0]):
                    ax.lines[0].set_ydata(z)
                    ax.relim()  # make sure all the data fits
                    ax.autoscale_view(True, True, True)
        else:
            extent = [times_Z[0], times_Z[-1], 1, len(Z_hat[0])]
            for k in range(n_atoms):
                im = axes_Z[k].imshow(Z_hat[k], cmap='hot', origin='lower',
                                      extent=extent,
                                      clim=(0.0, Z_hat.max()), aspect='auto')
            fig_Z.subplots_adjust(right=0.8)
            cbar_ax = fig_Z.add_axes([0.86, 0.10, 0.03, 0.8])
            fig_Z.colorbar(im, ax=None, cax=cbar_ax)

        fig.canvas.draw()
        fig_topo.canvas.draw()
        fig_Z.canvas.draw()
        plt.pause(.001)

    return callback


def plot_or_replot(data, axes=None, sfreq=1):
    """Given a list of axes and the ydata of each axis, cleanly update the plot

    Parameters
    ----------
    data : list of ydata
        Data to plot, or to update the axes lines.
    axes : list of Axes or None
        Axes to update.
    sfreq : float or None
        Value to compute the xlabel.
    """
    if axes is None:
        K = len(data)
        n_rows, n_cols = min(K, 5), max(1, K // 5)
        fig, axes = plt.subplots(n_rows, n_cols, squeeze=False).ravel()
    if axes[0].lines == []:
        color_cycle = itertools.cycle(COLORS)
        for ax, xk, color in zip(axes, data, color_cycle):
            ax.plot(np.arange(xk.shape[-1]) / sfreq, xk, c=color)
            ax.grid(True)
    else:
        for ax, xk in zip(axes, data):
            ax.lines[0].set_xdata(np.arange(xk.shape[-1]) / sfreq)
            ax.lines[0].set_ydata(xk)
            ax.relim()  # make sure all the data fits
            ax.autoscale_view(True, True, True)


def get_callback_csc(csc_kwargs, config=DEFAULT_CB, info={}):
    """Setup and return a callback for csc scripts

    Parameters
    ----------
    csc_kwargs : dict
        Parameters used to run the csc experiment. It will be hashed and used
        in the name of the files.
    info : dict
        Information on the signal.
    config : dict
        The key should be in {'atom', 'Xhat', 'Zhat', 'topo'}. and the value
        should be a configuration dictionary. The considered config are:
            'share': If True, share X and Y axis for all the axes.
            'info': Required for 'topo', should be the epoch.info.

    Return
    ------
    callback : callable
        A callback function that can be used to periodically plot and save
        the csc results.
    """
    csc_hash = positive_hash(json.dumps(csc_kwargs, sort_keys=True))
    csc_time = datetime.now().strftime('%d_%m_%y_%Hh%M')
    csc_script = get_calling_script()

    csc_name = "{}_csc_{}".format(csc_time, csc_hash)

    if not os.path.exists('save_exp'):
        os.mkdir('save_exp')
    save_dir = 'save_exp/{}'.format(csc_script)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Dump the arguments in a json to easily find introspect them
    with open('{}/{}.json'.format(save_dir, csc_name), 'w') as f:
        json.dump(csc_kwargs, f, sort_keys=True)

    n_atoms = csc_kwargs.get('n_atoms')
    n_times_atom = csc_kwargs.get('n_times_atom')

    # Extract plot information
    sfreq = info.get('sfreq', 1)
    t_min = info.get('t_min', 0)

    # Compute the shape of the subplot
    ncols = int(np.ceil(n_atoms / 5.))
    nrows = min(5, n_atoms)

    # Clean previous figures
    plt.close('all')

    figs = {}
    for name, conf in config.items():
        assert name in ['atom', 'Zhat', 'Xhat', 'topo', 'pobj']

        fname = "{} - {}".format(name, os.getpid())
        shared_axes = conf.get('share', True)
        width = 5
        nc, nr = ncols, nrows
        if name == 'topo':
            width = 3
        if name == 'pobj':
            nc = nr = 1
            width = 10
        f, axes = plt.subplots(nrows=nr, ncols=nc, num=fname,
                               figsize=(width * nc, 10), squeeze=False,
                               sharex=shared_axes, sharey=shared_axes)

        text = [PRINT_KWARGS.get(k, "{}={{}}".format(k)).format(v)
                for k, v in csc_kwargs.items() if k in PRINT_KWARGS]
        text = "\t".join(text)
        f.suptitle(text, fontsize=18, wrap=True)

        if name == 'pobj':
            axes[0, 0].set_xscale('log')
            axes[0, 0].set_yscale('log')

        figs[name] = (f, axes)
    for f, _ in figs.values():
        f.tight_layout()
        f.subplots_adjust(top=.9)
        f.canvas.draw()
    plt.pause(.001)

    def callback(X, uv_hat, Z_hat, pobj):

        n_channels = X.shape[1]

        if 'topo' in figs:
            axes = figs['topo'][1]
            for idx, ax in enumerate(axes.ravel()):
                ax.clear()
                mne.viz.plot_topomap(uv_hat[idx, :n_channels], info,
                                     axes=ax, show=False)
                ax.set_title('atom %d' % idx)

        if 'atom' in figs:
            axes = figs['atom'][1].ravel()
            if csc_kwargs.get('rank1', True):
                plot_or_replot(uv_hat[:, n_channels:], axes, sfreq=sfreq)
            else:
                i0 = uv_hat.std(axis=-1).argmax(axis=-1)
                plot_or_replot(
                    [uv_hat[k, i0[k]] for k in range(n_atoms)], axes)

        if 'Xhat' in figs:
            X_hat = [np.convolve(Zk, uvk[n_channels:])
                     for Zk, uvk in zip(Z_hat[:, 0], uv_hat)]
            axes = figs['Xhat'][1].ravel()
            plot_or_replot(X_hat, axes)

        if 'Zhat' in figs:
            axes = figs['Zhat'][1]
            plot_activations_density(Z_hat, n_times_atom, sfreq=sfreq,
                                     plot_activations=False, axes=axes,
                                     bandwidth='auto', t_min=t_min)

        if 'pobj' in figs:
            axes = figs['pobj'][1].ravel()
            cost = np.array([pobj])
            plot_or_replot(cost - np.min(cost) + 1e-6, axes)

        # Update and save the figures
        for fig_name, (f, _) in figs.items():
            f.canvas.draw()
            f.savefig('{}/{}_{}.png'.format(save_dir, csc_name, fig_name))
        plt.pause(.001)

    return callback
