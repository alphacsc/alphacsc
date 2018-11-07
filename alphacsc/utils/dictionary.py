import numpy as np
from scipy import signal


def get_D(uv_hat, n_channels):
    """Compute the rank 1 dictionary associated with the given uv

    Parameter
    ---------
    uv: array (n_atoms, n_channels + n_times_atom)
    n_channels: int
        number of channels in the original multivariate series

    Return
    ------
    D: array (n_atoms, n_channels, n_times_atom)
    """

    return uv_hat[:, :n_channels, None] * uv_hat[:, None, n_channels:]


def get_uv(D):
    """Project D on the space of rank 1 dictionaries

    Parameter
    ---------
    D: array (n_atoms, n_channels, n_times_atom)

    Return
    ------
    uv: array (n_atoms, n_channels + n_times_atom)
    """
    n_atoms, n_channels, n_times_atom = D.shape
    uv = np.zeros((n_atoms, n_channels + n_times_atom))
    for k, d in enumerate(D):
        U, s, V = np.linalg.svd(d)
        uv[k] = np.r_[U[:, 0], V[0]]
    return uv


def _patch_reconstruction_error(X, z, D):
    """Return the reconstruction error for each patches of size (P, L)."""
    n_trials, n_channels, n_times = X.shape
    if D.ndim == 2:
        n_times_atom = D.shape[1] - n_channels
    else:
        n_times_atom = D.shape[2]

    from .convolution import construct_X_multi
    X_hat = construct_X_multi(z, D, n_channels=n_channels)

    diff = (X - X_hat)**2
    patch = np.ones(n_times_atom)

    return np.sum([[np.convolve(patch, diff_ip, mode='valid')
                    for diff_ip in diff_i]
                   for diff_i in diff], axis=1)


def get_lambda_max(X, D_hat, sample_weights=None):

    # univariate case, add a dimension (n_channels = 1)
    if X.ndim == 2:
        X = X[:, None, :]
        D_hat = D_hat[:, None, :]
        if sample_weights is not None:
            sample_weights = sample_weights[:, None, :]

    n_trials, n_channels, n_times = X.shape

    if sample_weights is None:
        # no need for the last dimension if we only have ones
        if D_hat.ndim == 2:
            sample_weights = np.ones(n_trials)
        else:
            sample_weights = np.ones((n_trials, n_channels))

    # multivariate rank-1 case
    if D_hat.ndim == 2:
        return np.max([[
            np.convolve(
                np.dot(uv_k[:n_channels], X_i * W_i), uv_k[:n_channels - 1:-1],
                mode='valid') for X_i, W_i in zip(X, sample_weights)
        ] for uv_k in D_hat], axis=(1, 2))[:, None]

    # multivariate general case
    else:
        return np.max([[
            np.sum([
                np.correlate(D_kp, X_ip * W_ip, mode='valid')
                for D_kp, X_ip, W_ip in zip(D_k, X_i, W_i)
            ], axis=0) for X_i, W_i in zip(X, sample_weights)
        ] for D_k in D_hat], axis=(1, 2))[:, None]


def tukey_window(n_times_atom):
    window = signal.tukey(n_times_atom)
    window[0] = 1e-9
    window[-1] = 1e-9
    return window
