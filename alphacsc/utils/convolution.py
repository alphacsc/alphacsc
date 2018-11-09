"""Convolutional utilities for dictionary learning"""

# Authors: Thomas Moreau <thomas.moreau@inria.fr>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>

import numba
import numpy as np

from .. import cython_code
from .lil import get_z_shape, is_lil


def construct_X(z, ds):
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
    assert z.shape[0] == ds.shape[0]
    n_atoms, n_trials, n_times_valid = z.shape
    n_atoms, n_times_atom = ds.shape
    n_times = n_times_valid + n_times_atom - 1

    X = np.zeros((n_trials, n_times))
    for i in range(n_trials):
        X[i] = _choose_convolve(z[:, i], ds)
    return X


def construct_X_multi(z, D=None, n_channels=None):
    """
    Parameters
    ----------
    z : array, shape (n_trials, n_atoms, n_times_valid)
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
    n_trials, n_atoms, n_times_valid = get_z_shape(z)
    assert n_atoms == D.shape[0]
    if D.ndim == 2:
        n_times_atom = D.shape[1] - n_channels
    else:
        _, n_channels, n_times_atom = D.shape
    n_times = n_times_valid + n_times_atom - 1

    X = np.zeros((n_trials, n_channels, n_times))
    for i in range(n_trials):
        X[i] = _choose_convolve_multi(z[i], D=D, n_channels=n_channels)
    return X


def _sparse_convolve(z_i, ds):
    """Same as _dense_convolve, but use the sparsity of zi."""
    n_atoms, n_times_atom = ds.shape
    n_atoms, n_times_valid = z_i.shape
    n_times = n_times_valid + n_times_atom - 1
    Xi = np.zeros(n_times)
    for zik, dk in zip(z_i, ds):
        for nnz in np.where(zik != 0)[0]:
            Xi[nnz:nnz + n_times_atom] += zik[nnz] * dk
    return Xi


def _sparse_convolve_multi(z_i, ds):
    """Same as _dense_convolve, but use the sparsity of zi."""
    n_atoms, n_channels, n_times_atom = ds.shape
    n_atoms, n_times_valid = z_i.shape
    n_times = n_times_valid + n_times_atom - 1
    Xi = np.zeros(shape=(n_channels, n_times))
    for zik, dk in zip(z_i, ds):
        for nnz in np.where(zik != 0)[0]:
            Xi[:, nnz:nnz + n_times_atom] += zik[nnz] * dk
    return Xi


def _sparse_convolve_multi_uv(z_i, uv, n_channels):
    """Same as _dense_convolve, but use the sparsity of zi."""
    u = uv[:, :n_channels]
    v = uv[:, n_channels:]
    n_atoms, n_times_valid = z_i.shape
    n_atoms, n_times_atom = v.shape
    n_times = n_times_valid + n_times_atom - 1

    Xi = np.zeros(shape=(n_channels, n_times))
    for zik, uk, vk in zip(z_i, u, v):
        zik_vk = np.zeros(n_times)
        for nnz in np.where(zik != 0)[0]:
            zik_vk[nnz:nnz + n_times_atom] += zik[nnz] * vk

        Xi += zik_vk[None, :] * uk[:, None]

    return Xi


def _dense_convolve(z_i, ds):
    """Convolve z_i[k] and ds[k] for each atom k, and return the sum."""
    return sum([np.convolve(zik, dk) for zik, dk in zip(z_i, ds)], 0)


def _dense_convolve_multi(z_i, ds):
    """Convolve z_i[k] and ds[k] for each atom k, and return the sum."""
    return np.sum([[np.convolve(zik, dkp) for dkp in dk]
                   for zik, dk in zip(z_i, ds)], 0)


def _dense_convolve_multi_uv(z_i, uv, n_channels):
    """Convolve z_i[k] and uv[k] for each atom k, and return the sum."""
    u = uv[:, :n_channels]
    v = uv[:, n_channels:]
    n_atoms, n_times_valid = z_i.shape
    n_atoms, n_times_atom = v.shape
    n_times = n_times_valid + n_times_atom - 1

    Xi = np.zeros((n_channels, n_times))
    for zik, uk, vk in zip(z_i, u, v):
        zik_vk = np.convolve(zik, vk)
        Xi += zik_vk[None, :] * uk[:, None]

    return Xi


def _choose_convolve(z_i, ds):
    """Choose between _dense_convolve and _sparse_convolve with a heuristic
    on the sparsity of z_i, and perform the convolution.

    z_i : array, shape(n_atoms, n_times_valid)
        Activations
    ds : array, shape(n_atoms, n_times_atom)
        Dictionary
    """
    assert z_i.shape[0] == ds.shape[0]

    if np.sum(z_i != 0) < 0.01 * z_i.size:
        return _sparse_convolve(z_i, ds)
    else:
        return _dense_convolve(z_i, ds)


def _choose_convolve_multi(z_i, D=None, n_channels=None):
    """Choose between _dense_convolve and _sparse_convolve with a heuristic
    on the sparsity of z_i, and perform the convolution.

    z_i : array, shape(n_atoms, n_times_valid)
        Activations
    D : array
        The atoms. Can either be full rank with shape shape
        (n_atoms, n_channels, n_times_atom) or rank 1 with
        shape shape (n_atoms, n_channels + n_times_atom)
    n_channels : int
        Number of channels
    """
    assert z_i.shape[0] == D.shape[0]

    if is_lil(z_i):
        cython_code._assert_cython()
        if D.ndim == 2:
            return cython_code._fast_sparse_convolve_multi_uv(
                z_i, D, n_channels, compute_D=True)
        else:
            return cython_code._fast_sparse_convolve_multi(z_i, D)

    elif np.sum(z_i != 0) < 0.01 * z_i.size:
        if D.ndim == 2:
            return _sparse_convolve_multi_uv(z_i, D, n_channels)
        else:
            return _sparse_convolve_multi(z_i, D)

    else:
        if D.ndim == 2:
            return _dense_convolve_multi_uv(z_i, D, n_channels)
        else:
            return _dense_convolve_multi(z_i, D)


@numba.jit((numba.float64[:, :, :], numba.float64[:, :]), cache=True,
           nopython=True)
def numpy_convolve_uv(ztz, uv):
    """Compute the multivariate (valid) convolution of ztz and D

    Parameters
    ----------
    ztz: array, shape = (n_atoms, n_atoms, 2 * n_times_atom - 1)
        Activations
    uv: array, shape = (n_atoms, n_channels + n_times_atom)
        Dictionnary

    Returns
    -------
    G : array, shape = (n_atoms, n_channels, n_times_atom)
        Gradient
    """
    assert uv.ndim == 2
    n_times_atom = (ztz.shape[2] + 1) // 2
    n_atoms = ztz.shape[0]
    n_channels = uv.shape[1] - n_times_atom

    u = uv[:, :n_channels]
    v = uv[:, n_channels:][:, ::-1]

    G = np.zeros((n_atoms, n_channels, n_times_atom))
    for k0 in range(n_atoms):
        for k1 in range(n_atoms):
            for t in range(n_times_atom):
                G[k0, :, t] += (
                    np.sum(ztz[k0, k1, t:t + n_times_atom] * v[k1]) * u[k1, :])

    return G


def tensordot_convolve(ztz, D):
    """Compute the multivariate (valid) convolution of ztz and D

    Parameters
    ----------
    ztz: array, shape = (n_atoms, n_atoms, 2 * n_times_atom - 1)
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
        G[:, :, t] = np.tensordot(ztz[:, :, t:t + n_times_atom], D_revert,
                                  axes=([1, 2], [0, 2]))
    return G


def sort_atoms_by_explained_variances(D_hat, z_hat, n_channels):
    n_atoms = D_hat.shape[0]
    assert z_hat.shape[1] == n_atoms
    variances = np.zeros(n_atoms)
    for kk in range(n_atoms):
        variances[kk] = construct_X_multi(z_hat[:, kk:kk + 1],
                                          D_hat[kk:kk + 1],
                                          n_channels=n_channels).var()
    order = np.argsort(variances)[::-1]
    z_hat = z_hat[:, order, :]
    D_hat = D_hat[order, ...]
    return D_hat, z_hat
