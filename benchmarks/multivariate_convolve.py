import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import convolve
from scipy.signal import fftconvolve
from sklearn.externals.joblib import Memory
# from scipy.fftpack import next_fast_len

from numba import jit

memory = Memory(cachedir='', verbose=0)


def scipy_fftconvolve(ZtZ, D):
    """
    ZtZ.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
    D.shape = n_atoms, n_channels, n_times_atom
    """
    n_atoms, n_channels, n_times_atom = D.shape
    # TODO: try with zero padding to next_fast_len

    G = np.zeros(D.shape)
    for k0 in range(n_atoms):
        for k1 in range(n_atoms):
            for p in range(n_channels):
                G[k0, p] += fftconvolve(ZtZ[k0, k1], D[k1, p], mode='valid')
    return G


def numpy_convolve(ZtZ, D):
    """
    ZtZ.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
    D.shape = n_atoms, n_channels, n_times_atom
    """
    n_atoms, n_channels, n_times_atom = D.shape

    G = np.zeros(D.shape)
    for k0 in range(n_atoms):
        for k1 in range(n_atoms):
            for p in range(n_channels):
                G[k0, p] += convolve(ZtZ[k0, k1], D[k1, p], mode='valid')
    return G


@jit(nogil=True)
def dot_and_numba(ZtZ, D):
    """
    ZtZ.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
    D.shape = n_atoms, n_channels, n_times_atom
    """
    n_atoms, n_channels, n_times_atom = D.shape
    G = np.zeros(D.shape)
    for k0 in range(n_atoms):
        for k1 in range(n_atoms):
            for p in range(n_channels):
                for t in range(n_times_atom):
                    G[k0, p, t] += np.dot(ZtZ[k0, k1, t:t + n_times_atom],
                                          D[k1, p, ::-1])
    return G


@jit(nogil=True)
def sum_and_numba(ZtZ, D):
    """
    ZtZ.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
    D.shape = n_atoms, n_channels, n_times_atom
    """
    n_atoms, n_channels, n_times_atom = D.shape

    G = np.zeros(D.shape)
    for k0 in range(n_atoms):
        for p in range(n_channels):
            for t in range(n_times_atom):
                G[k0, p, t] += np.sum(
                    ZtZ[k0, :, t:t + n_times_atom] * D[:, p, ::-1])
    return G


def tensordot(ZtZ, D):
    """
    ZtZ.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
    D.shape = n_atoms, n_channels, n_times_atom
    """
    n_atoms, n_channels, n_times_atom = D.shape
    D = D[:, :, ::-1]

    G = np.zeros(D.shape)
    for t in range(n_times_atom):
        G[:, :, t] = np.tensordot(ZtZ[:, :, t:t + n_times_atom], D,
                                  axes=([1, 2], [0, 2]))
    return G


all_func = [
    numpy_convolve,
    scipy_fftconvolve,
    dot_and_numba,
    sum_and_numba,
    tensordot,
]


def test_equality():
    n_atoms, n_channels, n_times_atom = 5, 10, 15
    ZtZ = np.random.randn(n_atoms, n_atoms, 2 * n_times_atom - 1)
    D = np.random.randn(n_atoms, n_channels, n_times_atom)

    for func in all_func:
        assert np.allclose(func(ZtZ, D), all_func[0](ZtZ, D))


@memory.cache
def run_one(n_atoms, n_channels, n_times_atom, func):
    ZtZ = np.random.randn(n_atoms, n_atoms, 2 * n_times_atom - 1)
    D = np.random.randn(n_atoms, n_channels, n_times_atom)

    start = time.time()
    func(ZtZ, D)
    duration = time.time() - start
    return (n_atoms, n_channels, n_times_atom, func.__name__, duration)


def benchmark():
    n_atoms_range = [1, 2, 4, 8, 16]
    n_channels_range = [10, 20, 40, 80, 160]
    n_times_atom_range = [10, 20, 40, 80, 160]
    n_runs = (len(n_atoms_range) * len(n_channels_range) *
              len(n_times_atom_range) * len(all_func))

    k = 0
    results = []
    for n_atoms in n_atoms_range:
        for n_channels in n_channels_range:
            for n_times_atom in n_times_atom_range:
                for func in all_func:
                    print('%d/%d, %s' % (k, n_runs, func.__name__))
                    k += 1
                    results.append(
                        run_one(n_atoms, n_channels, n_times_atom, func))

    df = pd.DataFrame(results, columns=[
        'n_atoms', 'n_channels', 'n_times_atom', 'func', 'duration'
    ])
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    def plot(index, ax):
        pivot = df.pivot_table(columns='func', index=index, values='duration')
        pivot.plot(ax=ax)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('duration')

    plot('n_atoms', axes[0])
    plot('n_times_atom', axes[1])
    plot('n_channels', axes[2])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_equality()
    benchmark()
