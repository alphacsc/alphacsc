import time

import numba
import numpy as np
import pandas as pd
from scipy import sparse
from joblib import Memory
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean

from alphacsc.cython import _fast_compute_ztz_lil
from alphacsc.cython import _fast_compute_ztz_csr

memory = Memory(location='', verbose=0)


def naive_sum(z, n_times_atom):
    """
    ztz.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
    z.shape = n_atoms, n_trials, n_times - n_times_atom + 1)
    """
    n_atoms, n_trials, n_times_valid = z.shape

    ztz = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    t0 = n_times_atom - 1
    for k0 in range(n_atoms):
        for k in range(n_atoms):
            for i in range(n_trials):
                for t in range(n_times_atom):
                    if t == 0:
                        ztz[k0, k, t0] += (z[k0, i] * z[k, i]).sum()
                    else:
                        ztz[k0, k, t0 + t] += (
                            z[k0, i, :-t] * z[k, i, t:]).sum()
                        ztz[k0, k, t0 - t] += (
                            z[k0, i, t:] * z[k, i, :-t]).sum()
    return ztz


@numba.jit((numba.float64[:, :, :], numba.int64))
def sum_numba(z, n_times_atom):
    """
    ztz.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
    z.shape = n_atoms, n_trials, n_times - n_times_atom + 1)
    """
    n_atoms, n_trials, n_times_valid = z.shape

    ztz = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    t0 = n_times_atom - 1
    for k0 in range(n_atoms):
        for k in range(n_atoms):
            for i in range(n_trials):
                for t in range(n_times_atom):
                    if t == 0:
                        ztz[k0, k, t0] += (z[k0, i] * z[k, i]).sum()
                    else:
                        ztz[k0, k, t0 + t] += (
                            z[k0, i, :-t] * z[k, i, t:]).sum()
                        ztz[k0, k, t0 - t] += (
                            z[k0, i, t:] * z[k, i, :-t]).sum()
    return ztz


def tensordot(z, n_times_atom):
    """
    ztz.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
    z.shape = n_atoms, n_trials, n_times - n_times_atom + 1)
    """
    n_atoms, n_trials, n_times_valid = z.shape

    ztz = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    t0 = n_times_atom - 1

    axes = ([1, 2], [1, 2])

    for t in range(n_times_atom):
        if t == 0:
            ztz[:, :, t0] += np.tensordot(z, z, axes=axes)
        else:
            tmp = np.tensordot(z[:, :, :-t], z[:, :, t:], axes=axes)
            ztz[:, :, t0 + t] += tmp
            tmp = np.tensordot(z[:, :, t:], z[:, :, :-t], axes=axes)
            ztz[:, :, t0 - t] += tmp

    return ztz


all_func = [
    # naive_sum,
    sum_numba,
    tensordot,
    _fast_compute_ztz_lil,
    _fast_compute_ztz_csr,
]


def test_equality():
    n_atoms, n_trials, n_times_atom, n_times_valid = 5, 10, 20, 50
    z = np.random.randn(n_atoms, n_trials, n_times_valid)

    reference = all_func[0](z, n_times_atom)
    for func in all_func:

        if 'fast' in func.__name__:
            z_ = [sparse.lil_matrix(zi) for zi in np.swapaxes(z, 0, 1)]
        else:
            z_ = z

        assert np.allclose(func(z_, n_times_atom), reference)


@memory.cache
def run_one(n_atoms, sparsity, n_times_atom, n_times_valid, func):
    n_trials = 4
    z = sparse.random(n_atoms, n_trials * n_times_valid, density=sparsity)
    z = z.toarray().reshape(n_atoms, n_trials, n_times_valid)

    if 'fast' in func.__name__:
        z = [sparse.lil_matrix(zi) for zi in np.swapaxes(z, 0, 1)]

    start = time.time()
    func(z, n_times_atom)
    duration = time.time() - start
    label = func.__name__
    if label[0] == '_':
        label = label[1:]
    return (n_atoms, sparsity, n_times_atom, n_times_valid, label, duration)


def benchmark():
    n_atoms_range = [1, 3, 9]
    sparsity_range = [0.01, 0.03, 0.1, 0.3]
    n_times_atom_range = [8, 32, 128]
    n_times_valid_range = [1000, 30000, 10000]

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
