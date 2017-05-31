"""Convolutional dictionary learning"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour.10@gmail.com>
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
