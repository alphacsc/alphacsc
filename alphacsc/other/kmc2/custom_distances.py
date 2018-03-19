import numpy as np
from numpy.fft import rfft, irfft
from scipy.fftpack import next_fast_len  # noqa

from sklearn.metrics.pairwise import check_pairwise_arrays


def roll_invariant_euclidean_distances(X, Y=None, squared=False):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.
    The distance is the minimum of the euclidean distance over all rolls:

        dist(x, y) = min_\tau(||x(t) - y(t - \tau)||^2)

    Parameters
    ----------
    X : array, shape (n_samples_1, n_features)

    Y : array, shape (n_samples_2, n_features)

    squared : boolean
        Not used. Only for API compatibility.

    Returns
    -------
    distances : array, shape (n_samples_1, n_samples_2)

    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    X, Y = check_pairwise_arrays(X, Y)
    n_samples_1, n_features = X.shape
    n_samples_2, n_features = Y.shape

    X_norm = np.power(np.linalg.norm(X, axis=1), 2)
    Y_norm = np.power(np.linalg.norm(Y, axis=1), 2)

    # n_pads = 0
    # n_fft = next_fast_len(n_features + n_pads)
    n_fft = n_features  # not fast but otherwise the distance is wrong
    X_hat = rfft(X, n_fft, axis=1)
    Y_hat = rfft(Y, n_fft, axis=1).conj()

    # # broadcasting can have a huge memory cost
    # XY_hat = X_hat[:, None, :] * Y_hat[None, :, :]
    # XY = irfft(XY_hat, n_fft, axis=2).max(axis=2)
    # distances = X_norm[:, None] + Y_norm[None, :] - 2 * XY

    distances = np.zeros((n_samples_1, n_samples_2))
    for i in range(n_samples_1):
        for j in range(n_samples_2):
            XY = irfft(X_hat[i] * Y_hat[j], n_fft).max()
            distances[i, j] = X_norm[i] + Y_norm[j] - 2 * XY

    distances += 1e-12

    return distances
