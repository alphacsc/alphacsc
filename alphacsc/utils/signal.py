import numpy as np
from scipy.signal import hilbert


def fast_hilbert(array):
    n_points = array.shape[0]
    n_fft = next_power2(n_points)
    return hilbert(array, n_fft)[:n_points]


def next_power2(num):
    """Compute the smallest power of 2 >= to num.(float -> int)"""
    return 2 ** int(np.ceil(np.log2(num)))
