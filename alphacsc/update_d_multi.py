"""Convolutional dictionary learning"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>

import numpy as np
from scipy.signal import convolve

from .utils import construct_X_multi, check_consistent_shape


def _get_D(uv, n_chan):
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

    return np.array([np.outer(uvk[:n_chan], uvk[n_chan:]) for uvk in uv])


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
    grad_u = np.array([grad_dk.dot(uvk[n_chan:])
                       for grad_dk, uvk in zip(grad_d, uv)])
    grad_v = np.array([uvk[:n_chan].dot(grad_dk)
                       for grad_dk, uvk in zip(grad_d, uv)])
    return np.c_[grad_u, grad_v]


def prox_uv(uv):
    norm_uv = np.maximum(1, np.sqrt(np.sum(uv * uv, axis=1, keepdims=True)))
    return uv / norm_uv


def update_d(X, Z, u_hat0, v_hat0, ds_init=None, debug=False,
             max_iter=1000, step_size=.1, factr=1e-7, verbose=0):
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
    solver_kwargs : dict
        Parameters for the solver
    verbose : int
        Verbosity level.

    Returns
    -------
    d_hat : array, shape (k, n_times_atom)
        The atom to learn from the data.
    lambd_hats : float
        The dual variables
    """
    n_atoms, n_trials, n_times_valid = Z.shape
    _, n_chan, n_times = X.shape

    def objective(uv):
        D = _get_D(uv, n_chan)
        X_hat = construct_X_multi(Z, D)
        res = X - X_hat
        return .5 * np.sum(res * res)

    eps = np.finfo(np.float64).eps
    u_hat = u_hat0.copy()
    v_hat = v_hat0.copy()
    uv_hat = np.c_[u_hat, v_hat]
    f_last = objective(uv_hat)
    for ii in range(max_iter):
        uv_hat -= step_size * _gradient_uv(X, Z, uv_hat)
        uv_hat = prox_uv(uv_hat)
        f = objective(uv_hat)
        if (f_last - f) / max([abs(f), abs(f_last), 1]) <= factr * eps:
            break
        f_last = f
    if verbose > 1:
        print('%d iterations' % (ii + 1))
    return uv_hat