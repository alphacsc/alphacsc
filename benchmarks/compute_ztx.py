import time

import pandas as pd
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.externals.joblib import Memory
from scipy.stats.mstats import gmean

from alphacsc.cython import _fast_compute_ztx

memory = Memory(cachedir='', verbose=0)

def compute_ZtX(Z, X):
    """
    Z.shape = n_atoms, n_trials, n_times - n_times_atom + 1)
    X.shape = n_trials, n_channels, n_times
    ZtX.shape = n_atoms, n_channels, n_times_atom
    """
    n_atoms, n_trials, n_times_valid = Z.shape
    _, n_chan, n_times = X.shape
    n_times_atom = n_times - n_times_valid + 1

    ZtX = np.zeros((n_atoms, n_chan, n_times_atom))
    for k, n, t in zip(*Z.nonzero()):
        ZtX[k, :, :] += Z[k, n, t] * X[n, :, t:t + n_times_atom]

    return ZtX


all_func = [
    compute_ZtX,
    _fast_compute_ztx,
]


def test_equality():
    n_channels = 3
    n_atoms, n_trials, n_times_atom, n_times_valid = 5, 10, 20, 150
    n_times = n_times_valid + n_times_atom - 1
    X = np.random.randn(n_trials, n_channels, n_times)
    Z = np.random.randn(n_atoms, n_trials, n_times_valid)

    reference = all_func[0](Z, X)
    for func in all_func:

        if 'fast' in func.__name__:
            Z_ = [sparse.lil_matrix(zi) for zi in np.swapaxes(Z, 0, 1)]
        else:
            Z_ = Z

        assert np.allclose(func(Z_, X), reference)


@memory.cache
def run_one(n_atoms, sparsity, n_times_atom, n_times_valid, func):
    n_trials = 4
    Z = sparse.random(n_atoms, n_trials * n_times_valid, density=sparsity)
    Z = Z.toarray().reshape(n_atoms, n_trials, n_times_valid)

    n_channels = 3
    n_times = n_times_valid + n_times_atom - 1
    X = np.random.randn(n_trials, n_channels, n_times)

    if 'fast' in func.__name__:
        Z = [sparse.lil_matrix(zi) for zi in np.swapaxes(Z, 0, 1)]

    start = time.time()
    func(Z, X)
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
