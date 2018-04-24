import time

import pandas as pd
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.externals.joblib import Memory
from scipy.stats.mstats import gmean

from alphacsc.utils.compat import numba, jit
from alphacsc.cython import _fast_compute_ztz

memory = Memory(cachedir='', verbose=0)


def naive_sum(Z, n_times_atom):
    """
    ZtZ.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
    Z.shape = n_atoms, n_trials, n_times - n_times_atom + 1)
    """
    n_atoms, n_trials, n_times_valid = Z.shape

    ZtZ = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    t0 = n_times_atom - 1
    for k0 in range(n_atoms):
        for k in range(n_atoms):
            for i in range(n_trials):
                for t in range(n_times_atom):
                    if t == 0:
                        ZtZ[k0, k, t0] += (Z[k0, i] * Z[k, i]).sum()
                    else:
                        ZtZ[k0, k, t0 + t] += (
                            Z[k0, i, :-t] * Z[k, i, t:]).sum()
                        ZtZ[k0, k, t0 - t] += (
                            Z[k0, i, t:] * Z[k, i, :-t]).sum()
    return ZtZ


@jit((numba.float64[:, :, :], numba.int64))
def sum_numba(Z, n_times_atom):
    """
    ZtZ.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
    Z.shape = n_atoms, n_trials, n_times - n_times_atom + 1)
    """
    n_atoms, n_trials, n_times_valid = Z.shape

    ZtZ = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    t0 = n_times_atom - 1
    for k0 in range(n_atoms):
        for k in range(n_atoms):
            for i in range(n_trials):
                for t in range(n_times_atom):
                    if t == 0:
                        ZtZ[k0, k, t0] += (Z[k0, i] * Z[k, i]).sum()
                    else:
                        ZtZ[k0, k, t0 + t] += (
                            Z[k0, i, :-t] * Z[k, i, t:]).sum()
                        ZtZ[k0, k, t0 - t] += (
                            Z[k0, i, t:] * Z[k, i, :-t]).sum()
    return ZtZ


def tensordot(Z, n_times_atom):
    """
    ZtZ.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
    Z.shape = n_atoms, n_trials, n_times - n_times_atom + 1)
    """
    n_atoms, n_trials, n_times_valid = Z.shape

    ZtZ = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    t0 = n_times_atom - 1

    axes = ([1, 2], [1, 2])

    for t in range(n_times_atom):
        if t == 0:
            ZtZ[:, :, t0] += np.tensordot(Z, Z, axes=axes)
        else:
            tmp = np.tensordot(Z[:, :, :-t], Z[:, :, t:], axes=axes)
            ZtZ[:, :, t0 + t] += tmp
            tmp = np.tensordot(Z[:, :, t:], Z[:, :, :-t], axes=axes)
            ZtZ[:, :, t0 - t] += tmp

    return ZtZ


all_func = [
    naive_sum,
    sum_numba,
    tensordot,
    _fast_compute_ztz,
]


def test_equality():
    n_atoms, n_trials, n_times_atom, n_times_valid = 5, 10, 20, 150
    Z = np.random.randn(n_atoms, n_trials, n_times_valid)

    reference = all_func[0](Z, n_times_atom)
    for func in all_func:

        if 'fast' in func.__name__:
            Z_ = [sparse.lil_matrix(zi) for zi in np.swapaxes(Z, 0, 1)]
        else:
            Z_ = Z

        assert np.allclose(func(Z_, n_times_atom), reference)


@memory.cache
def run_one(n_atoms, sparsity, n_times_atom, n_times_valid, func):
    n_trials = 4
    Z = sparse.random(n_atoms, n_trials * n_times_valid, density=sparsity)
    Z = Z.toarray().reshape(n_atoms, n_trials, n_times_valid)

    if 'fast' in func.__name__:
        Z = [sparse.lil_matrix(zi) for zi in np.swapaxes(Z, 0, 1)]

    start = time.time()
    func(Z, n_times_atom)
    duration = time.time() - start
    label = func.__name__
    if label[0] == '_':
        label = label[1:]
    return (n_atoms, sparsity, n_times_valid, n_times_atom, label, duration)


def benchmark():
    n_atoms_range = [1, 4, 16]
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
