"""Convolutional dictionary learning"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>

import numpy as np
from scipy import optimize, signal
from joblib import Parallel, delayed

from .utils import check_consistent_shape
from .utils import _choose_convolve_multi


def update_z_multi(X, u, v, reg, z0=None, debug=False,
                   parallel=None, solver='l_bfgs', b_hat_0=None,
                   solver_kwargs=dict(), sample_weights=None):
    """Update Z using L-BFGS with positivity constraints

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data array
    u : array, shape (n_atoms, n_channels)
        The spatial atoms.
    v : array, shape (n_atoms, n_times_atom)
        The temporal atoms.
    reg : float
        The regularization constant
    z0 : None | array, shape (n_atoms, n_trials, n_times_valid)
        Init for z (can be used for warm restart).
    debug : bool
        If True, check the grad.
    parallel : instance of Parallel
        Context manager for running joblibs in a loop.
    solver : 'l_bfgs' | 'ista' | 'fista'
        The solver to use.
    b_hat_0 : array, shape ((n_times - n_times_atom + 1) * n_atoms)
        init vector for power_iteration with 'ista' solver
    solver_kwargs : dict
        Parameters for the solver
    sample_weights: array, shape (n_trials, n_channels, n_times)
        Weights applied on the cost function.
    verbose : int
        Verbosity level.

    Returns
    -------
    z : array, shape (n_atoms, n_trials, n_times - n_times_atom + 1)
        The true codes.
    """
    n_trials, n_channels, n_times = X.shape
    check_consistent_shape(X, sample_weights)
    n_atoms, n_times_atom = v.shape
    n_times_valid = n_times - n_times_atom + 1

    # now estimate the codes
    my_update_z = delayed(_update_z_multi_idx)
    if parallel is None:
        parallel = Parallel(n_jobs=1)

    zhats = parallel(
        my_update_z(X, u, v, reg, z0, i, debug, solver, b_hat_0, solver_kwargs,
                    sample_weights)
        for i in np.array_split(np.arange(n_trials), parallel.n_jobs))
    z_hat = np.vstack(zhats)

    z_hat2 = z_hat.reshape((n_trials, n_atoms, n_times_valid))
    z_hat2 = np.swapaxes(z_hat2, 0, 1)

    return z_hat2


def _fprime(u, v, zi, Xi=None, sample_weights=None, reg=None,
            return_func=False):
    """np.dot(D.T, X[i] - np.dot(D, zi)) + reg

    Parameters
    ----------
    u : array, shape (n_atoms, n_channels)
        The spatial atoms.
    v : array, shape (n_atoms, n_times_atom)
        The temporal atoms.
    zi : array, shape (n_atoms * n_times_valid)
        The activations
    Xi : array, shape (n_channels, n_times) or None
        The data array for one trial
    sample_weights : array, shape (n_times, n_channels) or None
        The sample weights for one trial
    reg : float or None
        The regularization constant
    return_func : boolean
        Returns also the objective function, used to speed up LBFGS solver

    Returns
    -------
    (func) : float
        The objective function
    grad : array, shape (n_atoms * n_times_valid)
        The gradient
    """
    n_atoms, n_channels = u.shape
    n_atoms, n_times_atom = v.shape
    zi_reshaped = zi.reshape((n_atoms, -1))

    ds = u[:, :, None] * v[:, None, :]
    n_atoms, n_channels, n_times_atom = ds.shape

    Dzi = _choose_convolve_multi(zi_reshaped, ds)
    # n_channels, n_times = Dzi.shape
    if Xi is not None:
        Dzi -= Xi

    if sample_weights is not None:
        if return_func:
            # preserve Dzi, we don't want to apply the weights twice in func
            wDzi = sample_weights * Dzi
        else:
            Dzi *= sample_weights
            wDzi = Dzi
    else:
        wDzi = Dzi

    if return_func:
        func = 0.5 * np.dot(wDzi.ravel(), Dzi.ravel())
        if reg is not None:
            func += reg * zi.sum()

    # multiply by the spatial filter u
    # n_atoms, n_channels = u.shape
    # n_channels, n_times = wDzi.shape
    uwDzi = np.dot(u, wDzi)

    # Now do the dot product with the transpose of D (D.T) which is
    # the conv by the reversed filter (keeping valid mode)
    # n_atoms, n_times = uwDzi.shape
    # n_atoms, n_times_atom = v.shape
    # n_atoms * n_times_valid = grad.shape
    grad = np.concatenate([
        signal.convolve(uwDzi_k, v_k[::-1], 'valid')
        for (uwDzi_k, v_k) in zip(uwDzi, v)
    ])
    # grad = -np.dot(D.T, X[i] - np.dot(D, zi))

    if reg is not None:
        grad += reg

    if return_func:
        return func, grad
    else:
        return grad


def _update_z_multi_idx(X, u, v, reg, z0, idxs, debug, solver="l_bfgs",
                        b_hat_0=None, solver_kwargs=dict(),
                        sample_weights=None):

    n_trials, n_channels, n_times = X.shape
    n_atoms, n_times_atom = v.shape
    n_times_valid = n_times - n_times_atom + 1
    bounds = [(0, None) for idx in range(n_atoms * n_times_valid)]

    zhats = []

    for i in idxs:
        if sample_weights is None:
            sample_weights_i = None
        else:
            sample_weights_i = sample_weights[i]

        def func_and_grad(zi):
            return _fprime(u, v, zi, Xi=X[i], reg=reg, return_func=True,
                           sample_weights=sample_weights_i)

        def grad_noreg(zi):
            return _fprime(u, v, zi, Xi=X[i], reg=None, return_func=False,
                           sample_weights=sample_weights_i)

        if z0 is None:
            f0 = np.zeros(n_atoms * n_times_valid)
        else:
            f0 = z0[:, i, :].reshape((n_atoms * n_times_valid))

        if debug:

            def pobj(zi):
                return func_and_grad(zi)[0]

            def fprime(zi):
                return func_and_grad(zi)[1]

            assert optimize.check_grad(pobj, fprime, f0) < 1e-2

        if solver == "l_bfgs":
            factr = solver_kwargs.get('factr', 1e15)  # default value
            zhat, f, d = optimize.fmin_l_bfgs_b(func_and_grad, f0, fprime=None,
                                                args=(), approx_grad=False,
                                                bounds=bounds, factr=factr)
        elif solver == "ista":
            raise NotImplementedError('Not adapted yet for n_channels')
        elif solver == "fista":
            raise NotImplementedError('Not adapted yet for n_channels')
        else:
            raise ValueError("Unrecognized solver %s. Must be 'ista', 'fista',"
                             " or 'l_bfgs'." % solver)

        zhats.append(zhat)
    return np.vstack(zhats)
