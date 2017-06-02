"""Convolutional dictionary learning"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>

import numpy as np
from scipy import signal


def construct_X(Z, ds):
    """
    Parameters
    ----------
    z : array, shape (n_atoms, n_trials, n_times_valid)
        The activations
    ds : array, shape (n_atoms, n_times_atom)
        The atom.

    Returns
    -------
    X : array, shape (n_trials, n_times)
    """
    assert Z.shape[0] == ds.shape[0]
    n_atoms, n_trials, n_times_valid = Z.shape
    n_atoms, n_times_atom = ds.shape
    n_times = n_times_valid + n_times_atom - 1
    X = np.zeros((n_trials, n_times))
    for i in range(Z.shape[1]):
        X[i] = _choose_convolve(Z[:, i, :], ds)
    return X


def _sparse_convolve(Zi, ds):
    """Same as _dense_convolve, but use the sparsity of zi."""
    n_atoms, n_times_atom = ds.shape
    n_atoms, n_times_valid = Zi.shape
    n_times = n_times_valid + n_times_atom - 1
    Xi = np.zeros(n_times)
    for zik, dk in zip(Zi, ds):
        for nnz in np.where(zik != 0)[0]:
            Xi[nnz:nnz + n_times_atom] += zik[nnz] * dk
    return Xi


def _dense_convolve(Zi, ds):
    """Convolve Zi[k] and ds[k] for each atom k, and return the sum."""
    return sum([signal.convolve(zik, dk)
               for zik, dk in zip(Zi, ds)], 0)


def _choose_convolve(Zi, ds):
    """Choose between _dense_convolve and _sparse_convolve with a heuristic
    on the sparsity of Zi, and perform the convolution.

    Zi : array, shape(n_atoms, n_times_valid)
        Activations
    ds : array, shape(n_atoms, n_times_atom)
        Dictionary
    """
    assert Zi.shape[0] == ds.shape[0]
    n_atoms, n_times_valid = Zi.shape
    n_atoms, n_times_atom = ds.shape
    if np.sum(Zi != 0) < 0.01 * Zi.size:
        return _sparse_convolve(Zi, ds)
    else:
        return _dense_convolve(Zi, ds)


def check_consistent_shape(*args):
    for array in args[1:]:
        if array is not None and array.shape != args[0].shape:
            raise ValueError('Incompatible shapes. Got '
                             '(%s != %s)' % (array.shape, args[0].shape))


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


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
