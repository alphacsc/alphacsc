import itertools

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

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
        bandwidth = n_times_atom / float(sfreq) / 2.
    elif bandwidth == 'grid':
        bandwidth = n_times_atom / float(sfreq) / 2. * np.logspace(-1, 1, 20)

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

        # compute the kernel density and plot it
        kde_x = kde_sklearn(time_instants[selection], time_instants,
                            bandwidth=bandwidth)
        ax.fill_between(time_instants, kde_x * n_elements, color=color,
                        alpha=0.5)

    return axes
