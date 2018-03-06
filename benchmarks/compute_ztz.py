import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals.joblib import Memory
from scipy.stats.mstats import gmean

from numba import jit

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


@jit()
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


all_func = [naive_sum, sum_numba, tensordot]


def test_equality():
    n_atoms, n_trials, n_times_atom, n_times_valid = 5, 10, 20, 150
    Z = np.random.randn(n_atoms, n_trials, n_times_valid)

    reference = all_func[0](Z, n_times_atom)
    for func in all_func:
        assert np.allclose(func(Z, n_times_atom), reference)


@memory.cache
def run_one(n_atoms, n_trials, n_times_atom, n_times_valid, func):
    Z = np.random.randn(n_atoms, n_trials, n_times_valid)

    start = time.time()
    func(Z, n_times_atom)
    duration = time.time() - start
    return (n_atoms, n_trials, n_times_valid, n_times_atom, func.__name__,
            duration)


def benchmark():
    n_atoms_range = [1, 4, 16]
    n_trials_range = [1, 4, 16]
    n_times_atom_range = [10, 40, 160]
    n_times_valid_range = [200, 800, 3200]

    n_runs = (len(n_atoms_range) * len(n_trials_range) * len(
        n_times_atom_range) * len(n_times_valid_range) * len(all_func))

    k = 0
    results = []
    for n_atoms in n_atoms_range:
        for n_trials in n_trials_range:
            for n_times_atom in n_times_atom_range:
                for n_times_valid in n_times_valid_range:
                    for func in all_func:
                        print('%d/%d, %s' % (k, n_runs, func.__name__))
                        k += 1
                        results.append(
                            run_one(n_atoms, n_trials, n_times_atom,
                                    n_times_valid, func))

    df = pd.DataFrame(results, columns=[
        'n_atoms', 'n_trials', 'n_times_atom', 'n_times_valid', 'func',
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
    plot('n_trials', axes[2])
    plot('n_times_valid', axes[3])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_equality()
    benchmark()
