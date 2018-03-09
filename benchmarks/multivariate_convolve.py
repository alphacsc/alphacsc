import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import convolve
from scipy.signal import fftconvolve
from sklearn.externals.joblib import Memory
from scipy.stats.mstats import gmean

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


def numpy_convolve_uv(ZtZ, uv):
    """
    ZtZ.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
    uv.shape = n_atoms, n_channels + n_times_atom
    """
    assert uv.ndim == 2
    n_times_atom = (ZtZ.shape[2] + 1) // 2
    n_atoms = ZtZ.shape[0]
    n_channels = uv.shape[1] - n_times_atom

    u = uv[:, :n_channels]
    v = uv[:, n_channels:]

    G = np.zeros((n_atoms, n_channels, n_times_atom))
    for k0 in range(n_atoms):
        for k1 in range(n_atoms):
            G[k0, :, :] += (convolve(ZtZ[k0, k1], v[k1], mode='valid')[None, :]
                            * u[k1, :][:, None])

    return G


all_func = [
    numpy_convolve,
    # scipy_fftconvolve,
    dot_and_numba,
    sum_and_numba,
    tensordot,
    numpy_convolve_uv,
]

try:
    from numba import jit

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

    all_func.extend([dot_and_numba, sum_and_numba])

except ImportError:
    pass


try:
    import tensorflow as tf
    raise ImportError()

    def tensorflow_conv(ZtZ, D):
        """
        ZtZ.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
        D.shape = n_atoms, n_channels, n_times_atom
        """
        n_atoms, n_channels, n_times_atom = D.shape
        with tf.Session() as session:
            tf_D = tf.placeholder(tf.float32,
                                  shape=(n_times_atom, n_atoms, n_channels))
            tf_ZtZ = tf.placeholder(tf.float32, shape=(ZtZ.shape))

            res = tf.nn.convolution(tf_ZtZ, tf_D, padding="VALID",
                                    data_format="NCW")
            return session.run(res, feed_dict={
                tf_D: np.moveaxis(D, -1, 0)[::-1], tf_ZtZ: ZtZ})

    all_func.append(tensorflow_conv)

except ImportError:
    pass

try:
    import torch

    def torch_conv(ZtZ, D):
        """
        ZtZ.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
        D.shape = n_atoms, n_channels, n_times_atom
        """
        D = D.swapaxes(0, 1)[:, :, ::-1].copy()
        filters = torch.autograd.Variable(torch.from_numpy(D))
        inputs = torch.autograd.Variable(torch.from_numpy(ZtZ))
        return torch.nn.functional.conv1d(inputs, filters).data.numpy()
        # set convolution filter to D

    all_func.append(torch_conv)

except ImportError:
    pass

# all_func = all_func[-2:]


def test_equality():
    n_atoms, n_channels, n_times_atom = 5, 10, 15
    ZtZ = np.random.randn(n_atoms, n_atoms, 2 * n_times_atom - 1)
    u = np.random.randn(n_atoms, n_channels)
    v = np.random.randn(n_atoms, n_times_atom)
    D = u[:, :, None] * v[:, None, :]

    reference = all_func[0](ZtZ, D)
    for func in all_func:
        if 'uv' in func.__name__:
            result = func(ZtZ, uv=np.hstack([u, v]))
        else:
            result = func(ZtZ, D=D)
        assert np.allclose(result, reference)


@memory.cache
def run_one(n_atoms, n_channels, n_times_atom, func):
    ZtZ = np.random.randn(n_atoms, n_atoms, 2 * n_times_atom - 1)

    if 'uv' in func.__name__:
        uv = np.random.randn(n_atoms, n_channels + n_times_atom)
        D = uv
    else:
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
        pivot = df.pivot_table(columns='func', index=index, values='duration',
                               aggfunc=gmean)
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
