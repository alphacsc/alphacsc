import numpy as np
from scipy.signal.windows import tukey


def get_D(uv_hat, n_channels):
    """Compute the rank 1 dictionary associated with the given uv

    Parameter
    ---------
    uv: array, shape (n_atoms, n_channels + n_times_atom)
    n_channels: int
        number of channels in the original multivariate series

    Return
    ------
    D: array, shape (n_atoms, n_channels, n_times_atom)
    """

    return uv_hat[:, :n_channels, None] * uv_hat[:, None, n_channels:]


def flip_uv(uv, n_channels):
    """Ensure the temporal pattern v peak is positive for each atom.

    If necessary, multiply both u and v by -1.

    Parameter
    ---------
    uv: array, shape (n_atoms, n_channels + n_times_atom)
        Rank1 dictionary which should be modified.
    n_channels: int
        number of channels in the original multivariate series

    Return
    ------
    uv: array, shape (n_atoms, n_channels + n_times_atom)
    """
    v = uv[:, n_channels:]
    index_array = np.argmax(np.abs(v), axis=1)
    peak_value = v[np.arange(len(v)), index_array]
    uv[peak_value < 0] *= -1
    return uv


def get_uv(D):
    """Project D on the space of rank 1 dictionaries

    Parameter
    ---------
    D: array, shape (n_atoms, n_channels, n_times_atom)

    Return
    ------
    uv: array, shape (n_atoms, n_channels + n_times_atom)
    """
    n_atoms, n_channels, n_times_atom = D.shape
    uv = np.zeros((n_atoms, n_channels + n_times_atom))
    for k, d in enumerate(D):
        U, s, V = np.linalg.svd(d)
        uv[k] = np.r_[U[:, 0], V[0]]
    return flip_uv(uv, n_channels)


def get_D_shape(D, n_channels):
    if D.ndim == 2:
        n_times_atom = D.shape[1] - n_channels
    else:
        if n_channels is None:
            n_channels = D.shape[1]
        else:
            assert n_channels == D.shape[1], (
                f"n_channels does not match D.shape: {D.shape}"
            )
        n_times_atom = D.shape[2]

    return (D.shape[0], n_channels, n_times_atom)


def _patch_reconstruction_error(X, z, D):
    """Return the reconstruction error for each patches of size (P, L)."""
    _, n_channels, _ = X.shape
    *_, n_times_atom = get_D_shape(D, n_channels)

    from .convolution import construct_X_multi
    X_hat = construct_X_multi(z, D, n_channels=n_channels)

    diff = (X - X_hat)**2
    patch = np.ones(n_times_atom)

    return np.sum([[np.convolve(patch, diff_ip, mode='valid')
                    for diff_ip in diff_i]
                   for diff_i in diff], axis=1)


def get_lambda_max(X, D_hat, sample_weights=None, q=1):
    """For each atom, compute the regularization parameter scaling.
    This value is usually defined as the smallest value for which 0 is
    a solution of the optimization problem.
    In order to avoid spurious values, this quantity can also be estimated
    as the q-quantile of the correlation between signal patches and the
    atom.

    Parameters
    ----------
    X : array, shape (n_trials, n_times) or
               shape (n_trials, n_channels, n_times)
        The data

    D_hat : array, shape (n_atoms, n_channels + n_times_atoms) or
                   shape (n_atoms, n_channels, n_times_atom)
        The atoms

    sample_weights : None | array, shape (n_trials, n_times) or
                                   shape (n_trials, n_channels, n_times)
        Weights to apply to the data.
        Defaults is None

    q : float
        Quantile to compute, which must be between 0 and 1 inclusive.
        Default is 1, i.e., the maximum is returned.

    Returns
    -------
    lambda_max : array, shape (n_atoms, 1)
    """
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
        return np.quantile([[
            np.convolve(
                np.dot(uv_k[:n_channels], X_i * W_i), uv_k[:n_channels - 1:-1],
                mode='valid') for X_i, W_i in zip(X, sample_weights)
        ] for uv_k in D_hat], axis=(1, 2), q=q)[:, None]

    # multivariate general case
    else:
        return np.quantile([[
            np.sum([
                np.correlate(D_kp, X_ip * W_ip, mode='valid')
                for D_kp, X_ip, W_ip in zip(D_k, X_i, W_i)
            ], axis=0) for X_i, W_i in zip(X, sample_weights)
        ] for D_k in D_hat], axis=(1, 2), q=q)[:, None]


class NoWindow():

    def window(self, d):
        return d

    def remove_window(self, d):
        return d

    def simple_window(self, d):
        return d

    def remove_simple_window(self, d):
        return d


class UVWindower(NoWindow):
    def __init__(self, n_times_atom, n_channels):
        self.n_channels = n_channels
        self.tukey_window = tukey_window(n_times_atom)[None, :]

    def window(self, d):
        d = d.copy()
        d[:, self.n_channels:] *= self.tukey_window
        return d

    def remove_window(self, d):
        d = d.copy()
        d[:, self.n_channels:] /= self.tukey_window
        return d

    def simple_window(self, d):
        return d * self.tukey_window

    def remove_simple_window(self, d):
        return d / self.tukey_window


class SimpleWindower(NoWindow):
    def __init__(self, n_times_atom):
        self.tukey_window = tukey_window(n_times_atom)[None, None, :]

    def window(self, d):
        return d * self.tukey_window

    def remove_window(self, d):
        return d / self.tukey_window


def tukey_window(n_times_atom):
    window = tukey(n_times_atom)
    window[0] = 1e-9
    window[-1] = 1e-9
    return window
