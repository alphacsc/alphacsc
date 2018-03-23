import itertools

import copy as cp
from mpl_toolkits.axes_grid1 import AxesGrid

import os
import mne
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


from .tools import get_calling_script, positive_hash

colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]


def kde_sklearn(x, x_grid, bandwidth, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    bandwidth = np.atleast_1d(bandwidth)

    n_samples, = x.shape
    if n_samples == 0:
        return np.zeros_like(x_grid)

    # Grid-search over bandwidth parameter
    if bandwidth.size > 1:
        param_grid = dict(bandwidth=bandwidth)
        gs = GridSearchCV(KernelDensity(**kwargs), param_grid=param_grid, cv=5)
        gs.fit(x[:, np.newaxis])

        print('selected bandwidth: %s' % (gs.best_params_, ))
        kde_skl = gs.best_estimator_
        bandwidth = kde_skl.bandwidth
    else:
        kde_skl = KernelDensity(bandwidth=bandwidth[0], **kwargs)
        kde_skl.fit(x[:, np.newaxis])

    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf) * bandwidth


def plot_activations_density(Z_hat, n_times_atom, sfreq=1., threshold=0.01,
                             bandwidth='auto', axes=None,
                             plot_activations=True):
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
    bandwidth : float, array of float, or in {'auto', 'grid'}
        Bandwidth (in sec) of the kernel
    axes : array of axes, or None
        Axes to plot into
    plot_activations : boolean
        If True, the significant activations are plotted as black dots
    """
    n_atoms, n_trials, n_times_valid = Z_hat.shape

    # sum activations over all trials
    Z_hat_sum = Z_hat.sum(axis=1)

    # normalize to maximum activation
    if Z_hat_sum.max() != 0:
        Z_hat_sum /= Z_hat_sum.max()

    if bandwidth == 'auto':
        bandwidth = n_times_atom / float(sfreq) / 4.
    elif bandwidth == 'grid':
        bandwidth = n_times_atom / float(sfreq) / 4. * np.logspace(-1, 1, 20)

    if axes is None:
        fig, axes = plt.subplots(n_atoms, num='density',
                                 figsize=(8, 2 + n_atoms * 3))

    color_cycle = itertools.cycle(colors)
    for ax, activations, color in zip(axes.ravel(), Z_hat_sum, color_cycle):
        ax.clear()
        time_instants = np.arange(n_times_valid) / float(sfreq)
        selection = activations > threshold
        n_elements = selection.sum()

        if n_elements == 0:
            ax.plot(time_instants, np.zeros_like(time_instants))
            continue

        # plot the activations as black dots
        if plot_activations:
            ax.plot(time_instants[selection],
                    activations[selection] / activations[selection].max(), '.',
                    color='k')

        # compute the kernel density and plot it
        kde_x = kde_sklearn(time_instants[selection], time_instants,
                            bandwidth=bandwidth)
        ax.fill_between(time_instants, kde_x * n_elements, color=color,
                        alpha=0.5)

    return axes


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
    nrows = 1 if n_trials > 1 else n_atoms
    fig_Z, axes_Z = plt.subplots(nrows=nrows, num='Z', figsize=(10, 8),
                                 sharex=True, sharey=True)
    fig_topo, axes_topo = plt.subplots(1, n_atoms_plot, figsize=(12, 3))

    if n_trials > 1:
        fig_Z.axes[0].axis('off')
        grid = AxesGrid(fig_Z, (0.1, 0.1, 0.8, 0.8),
                        nrows_ncols=(n_atoms, 1),
                        axes_pad=0.1,
                        share_all=True,
                        label_mode="L",
                        cbar_location="right",
                        cbar_mode="single",
                        cbar_size="2%"
                        )

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
            for k in range(n_atoms):
                im = grid[k].imshow(Z_hat[k], cmap='hot',
                                    clim=(0.0, Z_hat.max()))
                grid.cbar_axes[0].colorbar(im)

        fig.canvas.draw()
        fig_topo.canvas.draw()
        fig_Z.canvas.draw()
        plt.pause(.001)

    return callback


def get_callback_csc(csc_kwargs, sfreq=1, plot_topo=None):
    """Setup and return a callback for csc scripts

    Parameters
    ----------
    csc_kwargs : dict
        Parameters used to run the csc experiment. It will be hashed and used
        in the name of the files.
    sfreq : float
        Sampling frequency of the data, to correctly disply the time.
    plot_topo : Epoch.info or None
        If this parameter is used, plot the activation map of the learned
        dictionary.

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

    # Compute the shape of the subplot
    ncols = int(np.ceil(n_atoms / 5.))
    nrows = min(5, n_atoms)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, num='atoms',
                             figsize=(10, 8), sharex=True, sharey=True)
    fig_Z, axes_Z = plt.subplots(nrows=nrows, ncols=ncols, num='Zhat',
                                 figsize=(10, 8), sharex=True, sharey=True)
    fig_Xk, axes_Xk = plt.subplots(nrows=nrows, ncols=ncols, num="rec0",
                                   figsize=(15, 10))
    figs = [fig, fig_Xk, fig_Z]

    if plot_topo is not None:
        fig_topo, axes_topo = plt.subplots(nrows=ncols, ncols=nrows,
                                           num='topo', figsize=(12, 3 * ncols))
        figs += [fig_topo]

    for f in figs:
        f.tight_layout()
        f.canvas.draw()
    plt.pause(.001)

    def callback(X, uv_hat, Z_hat, reg):

        n_channels = X.shape[1]

        if plot_topo is not None:
            for idx, ax in enumerate(axes_topo.ravel()):
                ax.clear()
                mne.viz.plot_topomap(uv_hat[idx, :n_channels], plot_topo,
                                     axes=ax, show=False)
                ax.set_title('atom %d' % idx)

        if axes[0, 0].lines == []:
            for k, ax in enumerate(axes.ravel()):
                ax.plot(np.arange(n_times_atom) / sfreq,
                        uv_hat[k, n_channels:].T)
                ax.grid(True)
        else:
            for ax, uv in zip(axes.ravel(), uv_hat):
                ax.lines[0].set_ydata(uv[n_channels:])
                ax.relim()  # make sure all the data fits
                ax.autoscale_view(True, True, True)
        plot_activations_density(Z_hat, n_times_atom, sfreq=sfreq,
                                 plot_activations=False, axes=axes_Z,
                                 bandwidth='auto')

        X_hat = [np.convolve(Zk, uvk[n_channels:])
                 for Zk, uvk in zip(Z_hat[:, 0], uv_hat)]
        if axes_Xk[0, 0].lines == []:
            for k, ax in enumerate(axes_Xk.flatten()):
                ax.plot(np.arange(X_hat[k].shape[-1]) / sfreq,
                        X_hat[k])
                ax.grid(True)
        else:
            for ax, x in zip(axes_Xk.flatten(), X_hat):
                ax.lines[0].set_ydata(x)
                ax.relim()  # make sure all the data fits
                ax.autoscale_view(True, True, True)

        # Update and save the figures
        for f in figs:
            f.canvas.draw()
            fig_name = f.canvas.get_window_title()
            fig.savefig('{}/{}_{}.png'.format(save_dir, csc_name, fig_name))
        plt.pause(.001)

    return callback
