"""Convolutional dictionary learning"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>

import numpy as np
from scipy.signal import convolve

from .utils import construct_X_multi, _get_D


def _dense_transpose_convolve(Z, residual):
    """Convolve residual[i] with the transpose for each atom k, and return the sum

    Parameters
    ----------
    Z : array, shape (n_atoms, n_trials, n_times_valid)
    residual : array, shape (n_trials, n_chan, n_times)

    Return
    ------
    grad_D : array, shape (n_atoms, n_chan, n_times_atom)

    """
    return np.sum([[[convolve(res_ip, zik[::-1], mode='valid')  # n_times_atom
                     for res_ip in res_i]                       # n_chan
                    for zik, res_i in zip(zk, residual)]        # n_trials
                   for zk in Z], axis=1)                        # n_atoms


def _gradient_d(X, Z, D):
    residual = construct_X_multi(Z, D) - X
    return _dense_transpose_convolve(Z, residual)


def _gradient_uv(X, Z, uv):
    n_chan = X.shape[1]
    D = _get_D(uv, n_chan)
    grad_d = _gradient_d(X, Z, D)
    grad_u = (grad_d * uv[:, None, n_chan:]).sum(axis=2)
    grad_v = (grad_d * uv[:, :n_chan, None]).sum(axis=1)
    return np.c_[grad_u, grad_v]


def prox_uv(uv):
    norm_uv = np.maximum(1, np.linalg.norm(uv, axis=1)[:, None])
    return uv / norm_uv


def update_uv(X, Z, uv_hat0, debug=False, max_iter=300, step_size=1e-2,
             factr=1e-7, verbose=0):
    """Learn d's in time domain.

    Parameters
    ----------
    X : array, shape (n_trials, n_times)
        The data for sparse coding
    Z : array, shape (n_atoms, n_trials, n_times - n_times_atom + 1)
        The code for which to learn the atoms
    n_times_atom : int
        The shape of atoms.
    debug : bool
        If True, check grad.
    verbose : int
        Verbosity level.

    Returns
    -------
    uv_hat : array, shape (n_atoms, n_channels + n_times_atom)
        The atoms to learn from the data.
    """
    n_atoms, n_trials, n_times_valid = Z.shape
    _, n_chan, n_times = X.shape

    # XXX : step_size should not be hard coded but computed with power
    # method
    # XXX : computing objective for ealy stopping is brutal

    def objective(uv):
        D = _get_D(uv, n_chan)
        X_hat = construct_X_multi(Z, D)
        res = X - X_hat
        return .5 * np.sum(res * res)

    eps = np.finfo(np.float64).eps
    uv_hat = uv_hat0.copy()
    f_last = objective(uv_hat)
    for ii in range(max_iter):
        uv_hat -= step_size * _gradient_uv(X, Z, uv_hat)
        uv_hat = prox_uv(uv_hat)
        f = objective(uv_hat)
        if (f_last - f) / max([abs(f), abs(f_last), 1]) <= factr * eps:
            break
        f_last = f
    else:
        print('update_uv did not converge')
    if verbose > 1:
        print('%d iterations' % (ii + 1))
    return uv_hat
