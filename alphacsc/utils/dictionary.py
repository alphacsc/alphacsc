import numpy as np


from .convolution import construct_X_multi_uv


def _get_D(uv_hat, n_chan):
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


def _get_uv(D):
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


def _patch_reconstruction_error(X, Z, uv):
    """Return the reconstruction error for each patches of size (P, L)."""
    n_trials, n_channels, n_times = X.shape
    n_times_atom = uv.shape[1] - n_channels
    X_hat = construct_X_multi_uv(Z, uv, n_channels)

    diff = (X - X_hat)**2
    patch = np.ones(n_times_atom)

    return np.sum([[np.convolve(patch, diff_ip, mode='valid')
                    for diff_ip in diff_i]
                   for diff_i in diff], axis=1)
