"""Convolutional dictionary learning"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>

from __future__ import print_function
import time
import sys

import numpy as np
from joblib import Parallel

from .utils import check_random_state
from .loss_and_gradient import compute_X_and_objective_multi
from .update_z_multi import update_z_multi
from .update_d_multi import update_uv
from .init_dict import init_uv, get_max_error_dict
from .utils.profile_this import profile_this  # noqa


def learn_d_z_multi(X, n_atoms, n_times_atom, reg=0.1, n_iter=60, n_jobs=1,
                    solver_z='l_bfgs', solver_z_kwargs=dict(),
                    solver_d='alternate', solver_d_kwargs=dict(),
                    uv_constraint='separate', eps=1e-10,
                    uv_init=None, kmeans_params=dict(), stopping_pobj=None,
                    algorithm='batch', loss='l2',
                    loss_params=dict(gamma=.1, sakoe_chiba_band=10),
                    lmbd_max='fixed', verbose=10, callback=None,
                    random_state=None, name="DL"):
    """Learn atoms and activations using Convolutional Sparse Coding.

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data on which to perform CSC.
    n_atoms : int
        The number of atoms to learn.
    n_times_atom : int
        The support of the atom.
    reg : float
        The regularization parameter
    n_iter : int
        The number of coordinate-descent iterations.
    n_jobs : int
        The number of parallel jobs.
    solver_z : str
        The solver to use for the z update. Options are
        'l_bfgs' (default) | 'ista' | 'fista'
    solver_z_kwargs : dict
        Additional keyword arguments to pass to update_z_multi
    solver_d : str
        The solver to use for the d update. Options are
        'alternate' (default) | 'joint' | 'lbfgs'
    solver_d_kwargs : dict
        Additional keyword arguments to provide to update_d
    uv_constraint : str in {'joint', 'separate', 'box'}
        The kind of norm constraint on the atoms:
        If 'joint', the constraint is norm_2([u, v]) <= 1
        If 'separate', the constraint is norm_2(u) <= 1 and norm_2(v) <= 1
        If 'box', the constraint is norm_inf([u, v]) <= 1
    eps : float
        Stopping criterion. If the cost descent after a uv and a z update is
        smaller than eps, return.
    uv_init : array, shape (n_atoms, n_channels + n_times_atoms)
        The initial atoms.
    kmeans_params : dict
        Dictionnary of parameters for the kmeans init method.
    algorithm : 'batch' | 'greedy' | 'online'
        Dictionary learning algorithm.
    loss : 'l2' | 'dtw'
        Loss for the data-fit term. Either the norm l2 or the soft-DTW.
    loss_params : dict
        Parameters of the loss
    lmbd_max : 'fixed' | 'per_atom' | 'shared'
        If not fixed, adapt the regularization rate as a ratio of lambda_max.
    verbose : int
        The verbosity level.
    callback : func
        A callback function called at the end of each loop of the
        coordinate descent.
    random_state : int | None
        The random state.

    Returns
    -------
    pobj : list
        The objective function value at each step of the coordinate descent.
    times : list
        The cumulative time for each iteration of the coordinate descent.
    uv_hat : array, shape (n_atoms, n_channels + n_times_atom)
        The atoms to learn from the data.
    Z_hat : array, shape (n_atoms, n_trials, n_times_valid)
        The sparse activation matrix.
    """

    assert lmbd_max in ['fixed', 'per_atom', 'shared'], (
        "lmbd_max should be in {'fixed', 'per_atom', 'shared'}"
    )

    n_trials, n_chan, n_times = X.shape
    n_times_valid = n_times - n_times_atom + 1

    pobj = list()
    times = list()

    # initialization
    start = time.time()
    rng = check_random_state(random_state)

    uv_hat = init_uv(X, n_atoms, n_times_atom, uv_init=uv_init,
                     uv_constraint=uv_constraint, random_state=rng,
                     kmeans_params=kmeans_params)
    b_hat_0 = rng.randn(n_atoms * (n_chan + n_times_atom))
    times.append(time.time() - start)

    Z_hat = np.zeros((n_atoms, n_trials, n_times_valid))

    z_kwargs = dict(verbose=verbose)
    z_kwargs.update(solver_z_kwargs)

    def compute_z_func(X, Z_hat, uv_hat, reg=None, parallel=None):
        return update_z_multi(X, uv_hat, reg=reg, z0=Z_hat, parallel=parallel,
                              solver=solver_z, solver_kwargs=z_kwargs,
                              loss=loss, loss_params=loss_params)

    def obj_func(X, Z_hat, uv_hat, reg=None):
        return compute_X_and_objective_multi(X, Z_hat, uv_hat, reg=reg,
                                             uv_constraint=uv_constraint,
                                             loss=loss,
                                             loss_params=loss_params)

    d_kwargs = dict(verbose=verbose, eps=1e-8)
    d_kwargs.update(solver_d_kwargs)

    def compute_d_func(X, Z_hat, uv_hat):
        return update_uv(X, Z_hat, uv_hat0=uv_hat, b_hat_0=b_hat_0,
                         solver_d=solver_d, uv_constraint=uv_constraint,
                         loss=loss, loss_params=loss_params, **d_kwargs)

    end_iter_func = get_iteration_func(reg, eps, stopping_pobj, callback,
                                       lmbd_max)

    with Parallel(n_jobs=n_jobs) as parallel:
        if algorithm == 'batch':
            pobj, times, uv_hat, Z_hat = _batch_learn(
                X, uv_hat, Z_hat, compute_z_func, compute_d_func, obj_func,
                end_iter_func, n_iter=n_iter, n_jobs=n_jobs, verbose=verbose,
                random_state=random_state, parallel=parallel, reg=reg,
                lmbd_max=lmbd_max, name=name
            )
        elif algorithm == "greedy":
            raise NotImplementedError(
                "Algorithm greedy is not implemented yet.")
        elif algorithm == "online":
            raise NotImplementedError(
                "Algorithm online is not implemented yet.")
        else:
            raise NotImplementedError("Algorithm {} is not implemented to "
                                      "dictionary atoms.".format(algorithm))

        # recompute Z_hat with no regularization and keeping the support fixed
        Z_hat = update_z_multi(
            X, uv_hat, reg=0, z0=Z_hat, parallel=parallel, solver=solver_z,
            solver_kwargs=solver_z_kwargs, freeze_support=True, loss=loss,
            loss_params=loss_params)

    return pobj, times, uv_hat, Z_hat


def _batch_learn(X, uv_hat, Z_hat, compute_z_func, compute_d_func,
                 obj_func, end_iter_func, n_iter=100, n_jobs=1, verbose=0,
                 random_state=None, parallel=None, lmbd_max=False, reg=None,
                 name="batch"):

    pobj = list()
    times = list()

    # monitor cost function
    pobj.append(obj_func(X, Z_hat, uv_hat))

    n_channels = X.shape[1]
    reg_ = reg

    for ii in range(n_iter):  # outer loop of coordinate descent
        if verbose == 1:
            print('.', end='')
            sys.stdout.flush()
        if verbose > 1:
            print('[{}] CD iterations {} / {}'.format(name, ii, n_iter))

        if lmbd_max == 'per_atom':
            reg_ = reg * np.max([[
                np.convolve(np.dot(uv_k[:n_channels], X_i),
                            uv_k[:n_channels - 1:-1], mode='valid')
                for uv_k in uv_hat] for X_i in X], axis=(0, -1))[:, None]
        elif lmbd_max == 'shared':
            reg_ = reg * np.max([[
                np.convolve(np.dot(uv_k[:n_channels], X_i),
                            uv_k[:n_channels - 1:-1], mode='valid')
                for uv_k in uv_hat] for X_i in X])

        if verbose > 5:
            print('[{}] lambda = {:.3e}'.format(name, np.mean(reg_)))

        start = time.time()
        Z_hat = compute_z_func(X, Z_hat, uv_hat, reg=reg_, parallel=parallel)
        times.append(time.time() - start)

        if len(Z_hat.nonzero()[0]) == 0:
            import warnings
            warnings.warn("Regularization parameter `reg` is too large "
                          "and all the activations are zero. No atoms has"
                          " been learned.", UserWarning)
            break

        if verbose > 5:
            print("[{}] sparsity: {:.3e}".format(
                name, np.sum(Z_hat != 0) / Z_hat.size))

        # monitor cost function
        pobj.append(obj_func(X, Z_hat, uv_hat, reg=reg_))
        if verbose > 5:
            print('[{}] Objective (Z) : {:.3e}'.format(name, pobj[-1]))

        start = time.time()

        uv_hat = compute_d_func(X, Z_hat, uv_hat)
        times.append(time.time() - start)

        pobj.append(obj_func(X, Z_hat, uv_hat, reg=reg_))

        null_atom_indices = np.where(abs(Z_hat).sum(axis=(1, 2)) == 0)[0]
        if len(null_atom_indices) > 0:
            k0 = null_atom_indices[0]
            uv_hat[k0] = get_max_error_dict(X, Z_hat, uv_hat)[0]
            if verbose > 1:
                print('[{}] Resampled atom {}'.format(name, k0))

        if verbose > 5:
            print('[{}] Objective (d) : {:.3e}'.format(name, pobj[-1]))

        if end_iter_func(X, Z_hat, uv_hat, pobj):
            break

    return pobj, times, uv_hat, Z_hat


def get_iteration_func(reg, eps, stopping_pobj, callback, lmbd_max):
    def end_iteration(X, Z_hat, uv_hat, pobj):
        if callable(callback):
            callback(X, uv_hat, Z_hat, reg)

        # Only check that the cost is always going down when the regularization
        # parameter is fixed.
        if (pobj[-3] - pobj[-2] < eps * pobj[-1] and
                pobj[-2] - pobj[-1] < eps * pobj[-1] and
                lmbd_max == 'fixed'):
            if pobj[-3] - pobj[-2] < -eps:
                raise RuntimeError(
                    "The z update have increased the objective value.")
            if pobj[-2] - pobj[-1] < -eps:
                raise RuntimeError(
                    "The d update have increased the objective value.")
            return True

        if stopping_pobj is not None and pobj[-1] < stopping_pobj:
            return True
        return False

    return end_iteration
