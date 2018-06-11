"""Convolutional utilities for dictionary learning"""

# Authors: Thomas Moreau <thomas.moreau@inria.fr>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>

import numpy as np

from .compat import numba, jit
from .lil import get_Z_shape, is_lil
from ..cython import _fast_sparse_convolve_multi_uv
from ..cython import _fast_sparse_convolve_multi


def construct_X(Z, ds):
    """
    Parameters
    ----------
    z : array, shape (n_atoms, n_trials, n_times_valid)
        The activations
    ds : array, shape (n_atoms, n_times_atom)
        The atoms

    Returns
    -------
    X : array, shape (n_trials, n_times)
    """
    assert Z.shape[0] == ds.shape[0]
    n_atoms, n_trials, n_times_valid = Z.shape
    n_atoms, n_times_atom = ds.shape
    n_times = n_times_valid + n_times_atom - 1

    X = np.zeros((n_trials, n_times))
    for i in range(n_trials):
        X[i] = _choose_convolve(Z[:, i], ds)
    return X


def construct_X_multi(Z, D=None, n_channels=None):
    """
    Parameters
    ----------
    Z : array, shape (n_trials, n_atoms, n_times_valid)
        Can also be a list of n_trials LIL-sparse matrix of shape
            (n_atoms, n_times - n_times_atom + 1)
        The activations
    D : array
        The atoms. Can either be full rank with shape shape
        (n_atoms, n_channels, n_times_atom) or rank 1 with
        shape shape (n_atoms, n_channels + n_times_atom)
    n_channels : int
        Number of channels

    Returns
    -------
    X : array, shape (n_trials, n_channels, n_times)
    """
    n_trials, n_atoms, n_times_valid = get_Z_shape(Z)
    assert n_atoms == D.shape[0]
    if D.ndim == 2:
        n_times_atom = D.shape[1] - n_channels
    else:
        _, n_channels, n_times_atom = D.shape
    n_times = n_times_valid + n_times_atom - 1

    X = np.zeros((n_trials, n_channels, n_times))
    for i in range(n_trials):
        X[i] = _choose_convolve_multi(
            Z[i], D=D, n_channels=n_channels)
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


def _sparse_convolve_multi(Zi, ds):
    """Same as _dense_convolve, but use the sparsity of zi."""
    n_atoms, n_channels, n_times_atom = ds.shape
    n_atoms, n_times_valid = Zi.shape
    n_times = n_times_valid + n_times_atom - 1
    Xi = np.zeros(shape=(n_channels, n_times))
    for zik, dk in zip(Zi, ds):
        for nnz in np.where(zik != 0)[0]:
            Xi[:, nnz:nnz + n_times_atom] += zik[nnz] * dk
    return Xi


def _sparse_convolve_multi_uv(Zi, uv, n_channels):
    """Same as _dense_convolve, but use the sparsity of zi."""
    u = uv[:, :n_channels]
    v = uv[:, n_channels:]
    n_atoms, n_times_valid = Zi.shape
    n_atoms, n_times_atom = v.shape
    n_times = n_times_valid + n_times_atom - 1

    Xi = np.zeros(shape=(n_channels, n_times))
    for zik, uk, vk in zip(Zi, u, v):
        zik_vk = np.zeros(n_times)
        for nnz in np.where(zik != 0)[0]:
            zik_vk[nnz:nnz + n_times_atom] += zik[nnz] * vk

        Xi += zik_vk[None, :] * uk[:, None]

    return Xi


def _dense_convolve(Zi, ds):
    """Convolve Zi[k] and ds[k] for each atom k, and return the sum."""
    return sum([np.convolve(zik, dk)
               for zik, dk in zip(Zi, ds)], 0)


def _dense_convolve_multi(Zi, ds):
    """Convolve Zi[k] and ds[k] for each atom k, and return the sum."""
    return np.sum([[np.convolve(zik, dkp) for dkp in dk]
                   for zik, dk in zip(Zi, ds)], 0)


def _dense_convolve_multi_uv(Zi, uv, n_channels):
    """Convolve Zi[k] and uv[k] for each atom k, and return the sum."""
    u = uv[:, :n_channels]
    v = uv[:, n_channels:]
    n_atoms, n_times_valid = Zi.shape
    n_atoms, n_times_atom = v.shape
    n_times = n_times_valid + n_times_atom - 1

    Xi = np.zeros((n_channels, n_times))
    for zik, uk, vk in zip(Zi, u, v):
        zik_vk = np.convolve(zik, vk)
        Xi += zik_vk[None, :] * uk[:, None]

    return Xi


def _choose_convolve(Zi, ds):
    """Choose between _dense_convolve and _sparse_convolve with a heuristic
    on the sparsity of Zi, and perform the convolution.

    Zi : array, shape(n_atoms, n_times_valid)
        Activations
    ds : array, shape(n_atoms, n_times_atom)
        Dictionary
    """
    assert Zi.shape[0] == ds.shape[0]

    if np.sum(Zi != 0) < 0.01 * Zi.size:
        return _sparse_convolve(Zi, ds)
    else:
        return _dense_convolve(Zi, ds)


def _choose_convolve_multi(Zi, D=None, n_channels=None):
    """Choose between _dense_convolve and _sparse_convolve with a heuristic
    on the sparsity of Zi, and perform the convolution.

    Zi : array, shape(n_atoms, n_times_valid)
        Activations
    D : array
        The atoms. Can either be full rank with shape shape
        (n_atoms, n_channels, n_times_atom) or rank 1 with
        shape shape (n_atoms, n_channels + n_times_atom)
    n_channels : int
        Number of channels
    """
    assert Zi.shape[0] == D.shape[0]

    if is_lil(Zi):
        if D.ndim == 2:
            return _fast_sparse_convolve_multi_uv(Zi, D, n_channels,
                                                  compute_D=True)
        else:
            return _fast_sparse_convolve_multi(Zi, D)

    elif np.sum(Zi != 0) < 0.01 * Zi.size:
        if D.ndim == 2:
            return _sparse_convolve_multi_uv(Zi, D, n_channels)
        else:
            return _sparse_convolve_multi(Zi, D)

    else:
        if D.ndim == 2:
            return _dense_convolve_multi_uv(Zi, D, n_channels)
        else:
            return _dense_convolve_multi(Zi, D)


@jit((numba.float64[:, :, :], numba.float64[:, :]), cache=True, nopython=True)
def numpy_convolve_uv(ZtZ, uv):
    """Compute the multivariate (valid) convolution of ZtZ and D

    Parameters
    ----------
    ZtZ: array, shape = (n_atoms, n_atoms, 2 * n_times_atom - 1)
        Activations
    uv: array, shape = (n_atoms, n_channels + n_times_atom)
        Dictionnary

    Returns
    -------
    G : array, shape = (n_atoms, n_channels, n_times_atom)
        Gradient
    """
    assert uv.ndim == 2
    n_times_atom = (ZtZ.shape[2] + 1) // 2
    n_atoms = ZtZ.shape[0]
    n_channels = uv.shape[1] - n_times_atom

    u = uv[:, :n_channels]
    v = uv[:, n_channels:][:, ::-1]

    G = np.zeros((n_atoms, n_channels, n_times_atom))
    for k0 in range(n_atoms):
        for k1 in range(n_atoms):
            for t in range(n_times_atom):
                G[k0, :, t] += (np.sum(ZtZ[k0, k1, t:t + n_times_atom] * v[k1])
                                * u[k1, :])

    return G


def tensordot_convolve(ZtZ, D):
    """Compute the multivariate (valid) convolution of ZtZ and D

    Parameters
    ----------
    ZtZ: array, shape = (n_atoms, n_atoms, 2 * n_times_atom - 1)
        Activations
    D: array, shape = (n_atoms, n_channels, n_times_atom)
        Dictionnary

    Returns
    -------
    G : array, shape = (n_atoms, n_channels, n_times_atom)
        Gradient
    """
    n_atoms, n_channels, n_times_atom = D.shape
    D_revert = D[:, :, ::-1]

    G = np.zeros(D.shape)
    for t in range(n_times_atom):
        G[:, :, t] = np.tensordot(ZtZ[:, :, t:t + n_times_atom], D_revert,
                                  axes=([1, 2], [0, 2]))
    return G
