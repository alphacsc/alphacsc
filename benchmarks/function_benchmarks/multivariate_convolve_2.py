import time

import numpy as np
import pandas as pd
from scipy import sparse
from joblib import Memory
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean

from alphacsc.cython import _fast_sparse_convolve_multi
from alphacsc.cython import _fast_sparse_convolve_multi_uv

memory = Memory(location='', verbose=0)


def _dense_convolve_multi(z_i, ds):
    """Convolve z_i[k] and ds[k] for each atom k, and return the sum

    z_i : array, shape(n_atoms, n_times_valid)
        Activations
    ds : array, shape(n_atoms, n_channels, n_times_atom)
        Dictionary

    Returns
    -------
    res : array, shape(n_channels, n_times)
        Result of the convolution
    """
    return np.sum([[np.convolve(zik, dkp) for dkp in dk]
                   for zik, dk in zip(z_i, ds)], 0)


def _dense_convolve_multi_uv(z_i, uv, n_channels):
    """Convolve z_i[k] and uv[k] for each atom k, and return the sum

    z_i : array, shape(n_atoms, n_times_valid)
        Activations
    uv : array, shape(n_atoms, n_channels + n_times_atom)
        Dictionary
    n_channels : int
        Number of channels

    Returns
    -------
    Xi : array, shape(n_channels, n_times)
        Result of the convolution
    """
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


all_func = [
    _dense_convolve_multi,
    _dense_convolve_multi_uv,
    _sparse_convolve_multi,
    _sparse_convolve_multi_uv,
    _fast_sparse_convolve_multi,
    _fast_sparse_convolve_multi_uv,
]


def test_equality():
    n_atoms, n_channels, n_times_atom, n_times_valid = 5, 10, 15, 100
    z_i = np.random.randn(n_atoms, n_times_valid)
    u = np.random.randn(n_atoms, n_channels)
    v = np.random.randn(n_atoms, n_times_atom)
    D = u[:, :, None] * v[:, None, :]

    reference = all_func[0](z_i, D)
    for func in all_func:
        if 'uv' in func.__name__:
            kwargs = dict(uv=np.hstack([u, v]), n_channels=n_channels)
        else:
            kwargs = dict(ds=D)

        if 'fast' in func.__name__:
            z_i = sparse.lil_matrix(z_i)

        result = func(z_i, **kwargs)
        assert np.allclose(result, reference)


@memory.cache
def run_one(n_atoms, sparsity, n_times_atom, n_times_valid, func):
    n_channels = 4
    z_i = sparse.random(n_atoms, n_times_valid, density=sparsity, format='lil')

    if 'uv' in func.__name__:
        uv = np.random.randn(n_atoms, n_channels + n_times_atom)
        kwargs = dict(uv=uv, n_channels=n_channels)
    else:
        kwargs = dict(ds=np.random.randn(n_atoms, n_channels, n_times_atom))

    if 'fast' not in func.__name__:
        z_i = z_i.toarray()

    start = time.time()
    func(z_i, **kwargs)
    duration = time.time() - start
    return (n_atoms, sparsity, n_times_atom, n_times_valid, func.__name__,
            duration)


def benchmark():
    n_atoms_range = [1, 10, 100]
    sparsity_range = np.logspace(-4, -1, 5)
    n_times_atom_range = [10, 40, 160]
    n_times_valid_range = [200, 800, 3200]

    n_runs = (len(n_atoms_range) * len(sparsity_range) * len(
        n_times_atom_range) * len(n_times_valid_range) * len(all_func))

    k = 0
    results = []
    for n_atoms in n_atoms_range:
        for sparsity in sparsity_range:
            for n_times_atom in n_times_atom_range:
                for n_times_valid in n_times_valid_range:
                    for func in all_func:
                        print('%d/%d, %s' % (k, n_runs, func.__name__))
                        k += 1
                        results.append(
                            run_one(n_atoms, sparsity, n_times_atom,
                                    n_times_valid, func))

    df = pd.DataFrame(results, columns=[
        'n_atoms', 'sparsity', 'n_times_atom', 'n_times_valid', 'func',
        'duration'
    ])
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    def plot(index, ax):
        pivot = df.pivot_table(columns='func', index=index, values='duration',
                               aggfunc=gmean)
        pivot.plot(ax=ax)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('duration')

    plot('n_atoms', axes[0])
    plot('n_times_atom', axes[1])
    plot('sparsity', axes[2])
    plot('n_times_valid', axes[3])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_equality()
    benchmark()
