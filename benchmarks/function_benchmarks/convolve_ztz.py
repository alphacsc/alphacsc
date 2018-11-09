import time

import numba
import numpy as np
import pandas as pd
from joblib import Memory
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean


memory = Memory(location='', verbose=0)


@numba.jit((numba.float64[:, :, :], numba.float64[:, :]), cache=True)
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
    v = uv[:, n_channels:]

    G = np.zeros((n_atoms, n_channels, n_times_atom))
    for k0 in range(n_atoms):
        for k1 in range(n_atoms):
            G[k0, :, :] += (
                np.convolve(ztz[k0, k1], v[k1], mode='valid')[None, :]
                * u[k1, :][:, None])

    return G


@numba.jit((numba.float64[:, :, :], numba.float64[:, :]), cache=True,
           nopython=True)
def numpy_convolve_uv_nopython(ztz, uv):
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


all_func = [
    # naive_sum,
    numpy_convolve_uv,
    numpy_convolve_uv_nopython,
]


def test_equality():
    n_atoms, n_channels, n_times_atom = 5, 10, 50
    ztz = np.random.randn(n_atoms, n_atoms, 2 * n_times_atom - 1)
    uv = np.random.randn(n_atoms, n_channels + n_times_atom)

    reference = all_func[0](ztz, uv)
    for func in all_func:

        assert np.allclose(func(ztz, uv), reference)


@memory.cache
def run_one(n_atoms, n_channels, n_times_atom, func):

    ztz = np.random.randn(n_atoms, n_atoms, 2 * n_times_atom - 1)
    uv = np.random.randn(n_atoms, n_channels + n_times_atom)

    start = time.time()
    func(ztz, uv)
    duration = time.time() - start
    label = func.__name__
    if label[0] == '_':
        label = label[1:]
    return (n_atoms, n_channels, n_times_atom, label, duration)


def benchmark():
    n_atoms_range = [1, 3, 9]
    n_channels_range = [1, 25, 50, 100, 200]
    n_times_atom_range = [8, 32, 128]

    n_runs = (len(n_atoms_range) * len(n_channels_range) * len(
        n_times_atom_range) * len(all_func))

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
        pivot = df.pivot_table(columns='func', index=index, values='duration',
                               aggfunc=gmean)
        pivot.plot(ax=ax)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('duration')

    plot('n_atoms', axes[0])
    plot('n_times_atom', axes[1])
    plot('n_channels', axes[2])
    # plot('n_times_valid', axes[3])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_equality()
    benchmark()
