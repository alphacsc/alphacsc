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

from .utils import construct_X_multi_uv, check_random_state
from .update_z_multi import update_z_multi
from .update_d_multi import update_uv, prox_uv
from .init_dict import init_uv
from .profile_this import profile_this  # noqa


def objective(X, X_hat, Z_hat, reg):
    residual = X - X_hat
    obj = 0.5 * np.sum(residual * residual) + reg * Z_hat.sum()
    return obj


def compute_X_and_objective_multi(X, Z_hat, uv_hat, reg,
                                  feasible_evaluation=True,
                                  uv_constraint='joint'):
    """Compute X and return the value of the objective function

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data on which to perform CSC.
    Z_hat : array, shape (n_atoms, n_times - n_times_atom + 1)
        The sparse activation matrix.
    uv_hat : array, shape (n_atoms, n_channels + n_times_atom)
        The atoms to learn from the data.
    reg : float
        The regularization Parameters
    feasible_evaluation: boolean
        If feasible_evaluation is True, it first projects on the feasible set,
        i.e. norm(uv_hat) <= 1.
    uv_constraint : str in {'joint', 'separate'}
        The kind of norm constraint on the atoms:
        If 'joint', the constraint is norm([u, v]) <= 1
        If 'separate', the constraint is norm(u) <= 1 and norm(v) <= 1
    """
    n_chan = X.shape[1]

    if feasible_evaluation:
        Z_hat = Z_hat.copy()
        uv_hat = uv_hat.copy()
        # project to unit norm
        uv_hat, norm_uv = prox_uv(uv_hat, uv_constraint=uv_constraint,
                                  n_chan=n_chan, return_norm=True)
        # update z in the opposite way
        Z_hat *= norm_uv[:, None, None]

    X_hat = construct_X_multi_uv(Z_hat, uv_hat, n_chan)

    return objective(X, X_hat, Z_hat, reg)


# @profile_this
def learn_d_z_multi(X, n_atoms, n_times_atom, func_d=update_uv, reg=0.1,
                    n_iter=60, random_state=None, n_jobs=1, solver_z='l_bfgs',
                    solver_d='alternate', uv_constraint='separate',
                    solver_d_kwargs=dict(), solver_z_kwargs=dict(),
                    eps=1e-10, uv_init=None, verbose=10, callback=None,
                    stopping_pobj=None):
    """Learn atoms and activations using Convolutional Sparse Coding.

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data on which to perform CSC.
    n_atoms : int
        The number of atoms to learn.
    n_times_atom : int
        The support of the atom.
    func_d : callable
        The function to update the atoms.
    reg : float
        The regularization parameter
    n_iter : int
        The number of coordinate-descent iterations.
    random_state : int | None
        The random state.
    n_jobs : int
        The number of parallel jobs.
    solver_z : str
        The solver to use for the z update. Options are
        'l_bfgs' (default) | 'ista' | 'fista'
    solver_d : str
        The solver to use for the d update. Options are
        'alternate' (default) | 'joint' | 'lbfgs'
    uv_constraint : str in {'joint', 'separate', 'box'}
        The kind of norm constraint on the atoms:
        If 'joint', the constraint is norm_2([u, v]) <= 1
        If 'separate', the constraint is norm_2(u) <= 1 and norm_2(v) <= 1
        If 'box', the constraint is norm_inf([u, v]) <= 1
    solver_d_kwargs : dict
        Additional keyword arguments to provide to update_d
    solver_z_kwargs : dict
        Additional keyword arguments to pass to update_z_multi
    uv_init : array, shape (n_atoms, n_channels + n_times_atoms)
        The initial atoms.
    verbose : int
        The verbosity level.
    callback : func
        A callback function called at the end of each loop of the
        coordinate descent.

    Returns
    -------
    pobj : list
        The objective function value at each step of the coordinate descent.
    times : list
        The cumulative time for each iteration of the coordinate descent.
    uv_hat : array, shape (n_atoms, n_channels + n_times_atom)
        The atoms to learn from the data.
    Z_hat : array, shape (n_trials, n_atoms, n_times_valid)
        The sparse activation matrix.
    """
    n_trials, n_chan, n_times = X.shape
    n_times_valid = n_times - n_times_atom + 1

    rng = check_random_state(random_state)
    uv_hat = init_uv(X, n_atoms, n_times_atom, uv_init=uv_init,
                     uv_constraint=uv_constraint, random_state=rng)
    b_hat_0 = rng.randn(n_atoms * (n_chan + n_times_atom))

    pobj = list()
    times = list()

    Z_hat = np.zeros((n_atoms, n_trials, n_times_valid))

    pobj.append(compute_X_and_objective_multi(X, Z_hat, uv_hat, reg,
                uv_constraint=uv_constraint))
    times.append(0.)
    with Parallel(n_jobs=n_jobs) as parallel:
        for ii in range(n_iter):  # outer loop of coordinate descent
            if verbose == 1:
                print('.', end='')
                sys.stdout.flush()
            if verbose > 1:
                print('Coordinate descent loop %d / %d [n_jobs=%d]' %
                      (ii, n_iter, n_jobs))

            start = time.time()
            Z_hat = update_z_multi(
                X, uv_hat, reg=reg, z0=Z_hat, parallel=parallel,
                solver=solver_z, solver_kwargs=solver_z_kwargs)
            times.append(time.time() - start)

            if len(Z_hat.nonzero()[0]) == 0:
                import warnings
                warnings.warn("Regularization parameter `reg` is too large "
                              "and all the activations are zero. No atoms has"
                              " been learned.", UserWarning)
                break

            if verbose > 1:
                print("sparsity:", np.sum(Z_hat != 0) / Z_hat.size)

            # monitor cost function
            pobj.append(compute_X_and_objective_multi(X, Z_hat, uv_hat, reg,
                        uv_constraint=uv_constraint))
            if verbose > 1:
                print('[seed %s] Objective (Z) : %0.8f' % (random_state,
                                                           pobj[-1]))

            start = time.time()
            d_kwargs = dict(verbose=verbose, eps=1e-8)
            d_kwargs.update(solver_d_kwargs)

            uv_hat = func_d(X, Z_hat, uv_hat0=uv_hat, b_hat_0=b_hat_0,
                            solver_d=solver_d, uv_constraint=uv_constraint,
                            **d_kwargs)
            times.append(time.time() - start)

            pobj.append(compute_X_and_objective_multi(X, Z_hat, uv_hat, reg,
                        uv_constraint=uv_constraint))

            if verbose > 1:
                print('[seed %s] Objective (d) : %0.8f' % (random_state,
                                                           pobj[-1]))

            if callable(callback):
                callback(X, uv_hat, Z_hat, reg)

            if pobj[-3] - pobj[-2] < eps and pobj[-2] - pobj[-1] < eps:
                if pobj[-3] - pobj[-2] < -eps:
                    raise RuntimeError(
                        "The z update have increased the objective value.")
                if pobj[-2] - pobj[-1] < -eps:
                    raise RuntimeError(
                        "The d update have increased the objective value.")
                break

            if stopping_pobj is not None and pobj[-1] < stopping_pobj:
                break

    # recompute Z_hat with no regularization and keeping the support fixed
    Z_hat = update_z_multi(
        X, uv_hat, reg=0, z0=Z_hat, parallel=parallel,
        solver=solver_z, solver_kwargs=solver_z_kwargs, freeze_support=True)

    return pobj, times, uv_hat, Z_hat
