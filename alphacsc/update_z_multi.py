"""Convolutional dictionary learning"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>

import numpy as np
from numba import jit
from scipy import optimize
from joblib import Parallel, delayed

from .utils import _choose_convolve_multi, _get_D


def update_z_multi(X, uv, reg, z0=None, debug=False, parallel=None,
                   solver='l_bfgs', solver_kwargs=dict(),
                   freeze_support=False):
    """Update Z using L-BFGS with positivity constraints

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data array
    uv : array, shape (n_atoms, n_channels + n_times_atom)
        The spatial and temporal atoms.
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
    solver_kwargs : dict
        Parameters for the solver
    freeze_support : boolean
        If True, the support of z0 is frozen.

    Returns
    -------
    z : array, shape (n_atoms, n_trials, n_times - n_times_atom + 1)
        The true codes.
    """
    n_trials, n_channels, n_times = X.shape
    n_atoms, n_channels_n_times_atom = uv.shape
    n_times_atom = n_channels_n_times_atom - n_channels
    n_times_valid = n_times - n_times_atom + 1

    # now estimate the codes
    my_update_z = delayed(_update_z_multi_idx)
    if parallel is None:
        parallel = Parallel(n_jobs=1)

    zhats = parallel(
        my_update_z(X, uv, reg, z0, i, debug, solver, solver_kwargs,
                    freeze_support)
        for i in np.array_split(np.arange(n_trials), parallel.n_jobs))
    z_hat = np.vstack(zhats)

    z_hat2 = z_hat.reshape((n_trials, n_atoms, n_times_valid))
    z_hat2 = np.swapaxes(z_hat2, 0, 1)

    return z_hat2


def _fprime(uv, zi, Xi=None, reg=None, return_func=False):
    """

    Parameters
    ----------
    uv : array, shape (n_atoms, n_channels + n_times_atom)
        The spatial and temporal atoms
    zi : array, shape (n_atoms * n_times_valid)
        The activations
    Xi : array, shape (n_channels, n_times) or None
        The data array for one trial
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
    n_channels, n_times = Xi.shape
    n_atoms, n_channels_n_times_atom = uv.shape
    n_times_atom = n_channels_n_times_atom - n_channels
    zi_reshaped = zi.reshape((n_atoms, -1))

    ds = _get_D(uv, n_channels)
    n_atoms, n_channels, n_times_atom = ds.shape

    Dzi = _choose_convolve_multi(zi_reshaped, ds)
    # n_channels, n_times = Dzi.shape
    if Xi is not None:
        Dzi -= Xi

    if return_func:
        func = 0.5 * np.dot(Dzi.ravel(), Dzi.ravel())
        if reg is not None:
            func += reg * zi.sum()

    # multiply by the spatial filter u
    # n_atoms, n_channels = u.shape
    # n_channels, n_times = Dzi.shape
    uDzi = np.dot(uv[:, :n_channels], Dzi)

    # Now do the dot product with the transpose of D (D.T) which is
    # the conv by the reversed filter (keeping valid mode)
    # n_atoms, n_times = uDzi.shape
    # n_atoms, n_times_atom = v.shape
    # n_atoms * n_times_valid = grad.shape
    grad = np.concatenate([
        np.convolve(uDzi_k, v_k[::-1], 'valid')
        for (uDzi_k, v_k) in zip(uDzi, uv[:, n_channels:])
    ])

    if reg is not None:
        grad += reg

    if return_func:
        return func, grad
    else:
        return grad


def _update_z_multi_idx(X, uv, reg, z0, idxs, debug, solver="l_bfgs",
                        solver_kwargs=dict(), freeze_support=False):
    n_trials, n_channels, n_times = X.shape
    n_atoms, n_channels_n_times_atom = uv.shape
    n_times_atom = n_channels_n_times_atom - n_channels
    n_times_valid = n_times - n_times_atom + 1

    assert not (freeze_support and z0 is None), 'Impossible !'

    constants = {}
    zhats = []

    if solver == "gcd":
        constants['DtD'] = _compute_DtD(uv, n_channels)

    for i in idxs:

        def func_and_grad(zi):
            return _fprime(uv, zi, Xi=X[i], reg=reg, return_func=True)

        def grad_noreg(zi):
            return _fprime(uv, zi, Xi=X[i], reg=None, return_func=False)

        if z0 is None:
            f0 = np.zeros(n_atoms * n_times_valid)
        else:
            f0 = z0[:, i, :].reshape(n_atoms * n_times_valid)

        if freeze_support:
            bounds = [(0, 0) if z == 0 else (0, None) for z in f0]
        else:
            bounds = [(0, None) for z in f0]

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
                                                bounds=bounds, factr=factr,
                                                maxiter=1e6)
        elif solver == "ista":
            raise NotImplementedError('Not adapted yet for n_channels')
        elif solver == "fista":
            raise NotImplementedError('Not adapted yet for n_channels')
        elif solver == "gcd":
            f0 = f0.reshape(n_atoms, n_times_valid)
            zhat = _coordinate_descent_idx(X[i], uv, constants, reg=reg, z0=f0,
                                           **solver_kwargs)
            # raise NotImplementedError('Not implemented yet!')
        else:
            raise ValueError("Unrecognized solver %s. Must be 'ista', 'fista',"
                             " or 'l_bfgs'." % solver)

        zhats.append(zhat)
    return np.vstack(zhats)


def _support_least_square(X, uv, Z, debug=False):
    n_trials, n_chan, n_times = X.shape
    n_atoms, _, n_times_valid = Z.shape
    n_times_atom = n_times - n_times_valid + 1

    # Compute DtD
    DtD = _compute_DtD(uv, n_chan)
    t0 = n_times_atom - 1
    Z_hat = np.zeros(Z.shape)

    for idx in range(n_trials):
        Xi = X[idx]
        support_i = Z[:, idx].nonzero()
        n_support = len(support_i[0])
        if n_support == 0:
            continue
        rhs = np.zeros((n_support, n_support))
        lhs = np.zeros(n_support)

        for i, (k_i, t_i) in enumerate(zip(*support_i)):
            for j, (k_j, t_j) in enumerate(zip(*support_i)):
                dt = t_i - t_j
                if abs(dt) < n_times_atom:
                    rhs[i, j] = DtD[k_i, k_j, t0 + dt]
            aux_i = np.dot(uv[k_i, :n_chan], Xi[:, t_i:t_i + n_times_atom])
            lhs[i] = np.dot(uv[k_i, n_chan:], aux_i)

        # Solve the non-negative least-square with nnls
        z_star, a = optimize.nnls(rhs, lhs)
        if debug:
            f0 = _fprime(uv, Z[:, idx], X[idx], return_func=True, reg=0)[0]
        for i, (k_i, t_i) in enumerate(zip(*support_i)):
            Z_hat[k_i, idx, t_i] = z_star[i]

        if debug:
            f1 = _fprime(uv, Z[:, idx], X[idx], return_func=True, reg=0)[0]
            assert f1 <= f0

    return Z_hat


@jit
def _compute_DtD(uv, n_chan):
    """Compute the DtD matrix"""
    # TODO: benchmark the cross correlate function of numpy
    n_atoms, n_times_atom = uv.shape
    n_times_atom -= n_chan

    u = uv[:, :n_chan]
    v = uv[:, n_chan:]

    DtD = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    t0 = n_times_atom - 1
    for k0 in range(n_atoms):
        for k in range(n_atoms):
            for t in range(n_times_atom):
                if t == 0:
                    DtD[k0, k, t0] = np.dot(v[k0], v[k])
                else:
                    DtD[k0, k, t0 + t] = np.dot(v[k0, :-t], v[k, t:])
                    DtD[k0, k, t0 - t] = np.dot(v[k0, t:], v[k, :-t])
    DtD *= np.dot(u, u.T)[:, :, None]
    return DtD


def _coordinate_descent_idx(Xi, uv, constants, reg, z0=None, max_iter=1000,
                            tol=1e-5, strategy='greedy', n_seg='auto',
                            debug=False, verbose=0):
    """Compute the coding signal associated to Xi with coordinate descent.

    Parameters
    ----------
    Xi : array, shape (n_channels, n_times)
        The signal to encode.
    constants : dict
        Constants containing DtD to speedup computation
    z0 : array, shape (n_atoms, n_time_valid)
        Initial estimate of the coding signal, to warm start the algorithm.
    tol : float
        Tolerance for the stopping criterion of the algorithm
    max_iter : int
        Maximal number of iterations run by the algorithm
    strategy : str in {'greedy' | 'random'}
        Strategy to select the updated coordinate in the CD algorithm.
    n_seg : int or 'auto'
        Number of segments used to divide the coding signal. The updates are
        performed successively on each of these segments.
    """
    n_chan, n_times = Xi.shape
    n_atoms, n_times_atom = uv.shape
    n_times_atom -= n_chan
    n_times_valid = n_times - n_times_atom + 1
    t0 = n_times_atom - 1

    if z0 is None:
        z_hat = np.zeros((n_atoms, n_times_valid))
    else:
        z_hat = z0.copy()

    if n_seg == 'auto':
        n_seg = max(n_times_valid // (2 * n_times_atom), 1)

    n_times_seg = n_times_valid // n_seg + 1

    def objective(zi):
        ds = _get_D(uv, n_chan)

        Dzi = _choose_convolve_multi(zi, ds)
        Dzi -= Xi
        func = 0.5 * np.dot(Dzi.ravel(), Dzi.ravel())
        func += reg * zi.sum()
        return func

    DtD = constants["DtD"]
    norm_Dk = np.array([DtD[k, k, t0] for k in range(n_atoms)]).reshape(-1, 1)
    if debug:
        pobj = [objective(z_hat)]

    # Init beta with -DtX
    beta = _fprime(uv, z_hat.ravel(), Xi=Xi, reg=None, return_func=False)
    beta = beta.reshape(n_atoms, n_times_valid)
    for k, t in zip(*z_hat.nonzero()):
        beta[k, t] -= z0[k, t] * norm_Dk[k]  # np.sum(DtD[k, k, t0])
    z_opt = np.maximum(-beta - reg, 0) / norm_Dk

    dZs = 2 * tol * np.ones(n_seg)
    i_seg, t_start_seg = 0, 0
    t_end_seg = n_times_seg
    for ii in range(max_iter):
        # Pick a coordinate to update
        if strategy == 'random':
            raise NotImplementedError()
        elif strategy == 'greedy':
            i0 = 0
            i0 = np.argmax(np.abs(z_hat[:, t_start_seg:t_end_seg] -
                                  z_opt[:, t_start_seg:t_end_seg]))
            n_times_current = min(n_times_seg, n_times_valid - t_start_seg)
        else:
            raise ValueError('The coordinate selection method should be in '
                             "{'greedy' | 'random'}. Got {}.".format(strategy))

        k0, t0 = np.unravel_index(i0, (n_atoms, n_times_current))
        t0 += t_start_seg
        dz = z_hat[k0, t0] - z_opt[k0, t0]
        dZs[i_seg] = abs(dz)

        # Update the selected coordinate and beta if the update is greater than
        # the convergence tolerance.
        if abs(dz) > tol:
            z_hat[k0, t0] = z_opt[k0, t0]

            beta_i0 = beta[k0, t0]
            offset = max(0, n_times_atom - t0 - 1)
            t_start = max(0, t0 - n_times_atom + 1)
            ll = min(t0 + n_times_atom, n_times_valid) - t_start
            beta[:, t_start:t0 + n_times_atom] -= DtD[:, k0, offset:offset + ll] * dz
            beta[k0, t0] = beta_i0
            z_opt[:, t_start:t0 + n_times_atom] = np.maximum(
                -beta[:, t_start:t0 + n_times_atom] - reg, 0) / norm_Dk

        if debug:
            pobj.append(objective(z_hat))

        if dZs.max() <= tol:
            break

        i_seg += 1
        t_start_seg += n_times_seg
        t_end_seg += n_times_seg
        if i_seg >= n_seg:
            i_seg = 0
            t_start_seg = 0
            t_end_seg = n_times_seg

    else:
        if verbose > 1:
            print('update z [cd] did not converge')
    if verbose > 1:
        print('[CD] computed %d iterations' % (ii + 1))

    if debug:
        return z_hat, pobj
    return z_hat
