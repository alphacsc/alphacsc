import numpy as np


from .convolution import construct_X_multi


def get_D(uv_hat, n_chan):
    """Compute the rank 1 dictionary associated with the given uv

    Parameter
    ---------
    uv: array (n_atoms, n_chan + n_times_atom)
    n_chan: int
        number of channels in the original multivariate series

    Return
    ------
    D: array (n_atoms, n_chan, n_times_atom)
    """

    return uv_hat[:, :n_chan, None] * uv_hat[:, None, n_chan:]


def get_uv(D):
    """Project D on the space of rank 1 dictionaries

    Parameter
    ---------
    D: array (n_atoms, n_chan, n_times_atom)

    Return
    ------
    uv: array (n_atoms, n_chan + n_times_atom)
    """
    n_atoms, n_chan, n_times_atom = D.shape
    uv = np.zeros((n_atoms, n_chan + n_times_atom))
    for k, d in enumerate(D):
        U, s, V = np.linalg.svd(d)
        uv[k] = np.r_[U[:, 0], V[0]]
    return uv


def _patch_reconstruction_error(X, Z, D):
    """Return the reconstruction error for each patches of size (P, L)."""
    n_trials, n_channels, n_times = X.shape
    if D.ndim == 2:
        n_times_atom = D.shape[1] - n_channels
    else:
        n_times_atom = D.shape[2]

    X_hat = construct_X_multi(Z, D, n_channels=n_channels)

    diff = (X - X_hat)**2
    patch = np.ones(n_times_atom)

    return np.sum([[np.convolve(patch, diff_ip, mode='valid')
                    for diff_ip in diff_i]
                   for diff_i in diff], axis=1)


def get_lambda_max(X, D_hat):
    n_channels = X.shape[1]

    if D_hat.ndim == 2:
        return np.max([[
            np.convolve(np.dot(uv_k[:n_channels], X_i),
                        uv_k[:n_channels - 1:-1], mode='valid')
            for X_i in X] for uv_k in D_hat], axis=(1, 2))[:, None]
    else:
        return np.max([[
            np.sum([np.correlate(D_kp, X_ip, mode='valid')
                    for D_kp, X_ip in zip(D_k, X_i)], axis=0)
            for X_i in X] for D_k in D_hat], axis=(1, 2))[:, None]
