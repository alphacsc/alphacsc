"""Convolutional dictionary learning"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>

import numpy as np
from scipy import optimize

from .utils.lil import get_Z_shape, is_list_of_lil
from .utils.optim import fista, power_iteration
from .utils.convolution import numpy_convolve_uv
from .utils.compute_constants import compute_ZtZ, compute_ZtX
from .cython import _fast_compute_ZtZ, _fast_compute_ZtX

from .loss_and_gradient import compute_objective, compute_X_and_objective_multi
from .loss_and_gradient import gradient_uv, gradient_d


def prox_uv(uv, uv_constraint='joint', n_chan=None, return_norm=False):
    if uv_constraint == 'joint':
        norm_uv = np.maximum(1, np.linalg.norm(uv, axis=1))
        uv /= norm_uv[:, None]

    elif uv_constraint == 'separate':
        assert n_chan is not None
        norm_u = np.maximum(1, np.linalg.norm(uv[:, :n_chan], axis=1))
        norm_v = np.maximum(1, np.linalg.norm(uv[:, n_chan:], axis=1))

        uv[:, :n_chan] /= norm_u[:, None]
        uv[:, n_chan:] /= norm_v[:, None]
        norm_uv = norm_u * norm_v

    elif uv_constraint == 'box':
        assert n_chan is not None
        norm_u = np.maximum(1, np.max(uv[:, :n_chan], axis=1))
        norm_v = np.maximum(1, np.max(uv[:, n_chan:], axis=1))

        uv[:, :n_chan] /= norm_u[:, None]
        uv[:, n_chan:] /= norm_v[:, None]
        norm_uv = norm_u * norm_v
    else:
        raise ValueError('Unknown uv_constraint: %s.' % (uv_constraint, ))

    if return_norm:
        return uv, norm_uv
    else:
        return uv


def prox_d(D, return_norm=False):
    norm_d = np.maximum(1, np.linalg.norm(D, axis=(1, 2)))
    D /= norm_d[:, None, None]

    if return_norm:
        return D, norm_d
    else:
        return D


def update_uv(X, Z, uv_hat0, b_hat_0=None, debug=False, max_iter=300, eps=None,
              solver_d='alternate', momentum=False, uv_constraint='separate',
              loss='l2', loss_params=dict(), verbose=0):
    """Learn d's in time domain.

    Parameters
    ----------
    X : array, shape (n_trials, n_times)
        The data for sparse coding
    Z : array, shape (n_atoms, n_trials, n_times - n_times_atom + 1)
        Can also be a list of n_trials LIL-sparse matrix of shape
            (n_atoms, n_times - n_times_atom + 1)
        The code for which to learn the atoms
    uv_hat0 : array, shape (n_atoms, n_channels + n_times_atom)
        The initial atoms.
    b_hat_0 : array, shape (n_atoms * (n_channels + n_times_atom))
        Init eigen-vector vector used in power_iteration, used in warm start.
    debug : bool
        If True, return the cost at each iteration.
    momentum : bool
        If True, use an accelerated version of the proximal gradient descent.
    uv_constraint : str in {'joint', 'separate', 'box'}
        The kind of norm constraint on the atoms:
        If 'joint', the constraint is norm_2([u, v]) <= 1
        If 'separate', the constraint is norm_2(u) <= 1 and norm_2(v) <= 1
        If 'box', the constraint is norm_inf([u, v]) <= 1
    solver_d : str in {'alternate', 'joint', 'lbfgs'}
        The type of solver to update d:
        If 'alternate', the solver alternates between u then v
        If 'joint', the solver jointly optimize uv with a line search
        If 'lbfgs', the solver uses lbfgs with box constraints
    loss : str in {'l2' | 'dtw' | 'whitening'}
        The data-fit
    loss_params : dict
        Parameters of the loss
    verbose : int
        Verbosity level.

    Returns
    -------
    uv_hat : array, shape (n_atoms, n_channels + n_times_atom)
        The atoms to learn from the data.
    """
    n_atoms, n_trials, n_times_valid = get_Z_shape(Z)
    _, n_chan, n_times = X.shape

    if solver_d == 'lbfgs':
        msg = "L-BFGS sovler only works with box constraints"
        assert uv_constraint == 'box', msg
    elif solver_d == 'alternate':
        msg = "alternate solver should be used with separate constraints"
        assert uv_constraint == 'separate', msg

    if loss == 'l2':
        constants = _get_d_update_constants(X, Z)
    else:
        constants = None

    def objective(uv):
        if loss == 'l2':
            return compute_objective(D=uv, constants=constants)
        return compute_X_and_objective_multi(X, Z, D_hat=uv, loss=loss,
                                             loss_params=loss_params,
                                             feasible_evaluation=False)

    if solver_d in ['joint', 'fista']:
        # use FISTA on joint [u, v], with an adaptive step size

        def grad(uv):
            return gradient_uv(uv=uv, X=X, Z=Z, constants=constants, loss=loss,
                               loss_params=loss_params)

        def prox(uv):
            return prox_uv(uv, uv_constraint=uv_constraint, n_chan=n_chan)

        uv_hat, pobj = fista(objective, grad, prox, None, uv_hat0, max_iter,
                             verbose=verbose, momentum=momentum, eps=eps,
                             adaptive_step_size=True, debug=debug,
                             name="Update uv")

    elif solver_d in ['alternate', 'alternate_adaptive']:
        # use FISTA on alternate u and v

        adaptive_step_size = (solver_d == 'alternate_adaptive')

        uv_hat = uv_hat0.copy()
        u_hat, v_hat = uv_hat[:, :n_chan], uv_hat[:, n_chan:]

        def prox(u):
            u /= np.maximum(1., np.linalg.norm(u, axis=1))[:, None]
            return u

        for jj in range(1):
            # ---------------- update u
            def obj(u):
                uv = np.c_[u, v_hat]
                return objective(uv)

            def grad_u(u):
                uv = np.c_[u, v_hat]
                grad_d = gradient_d(uv, X=X, Z=Z, constants=constants,
                                    loss=loss, loss_params=loss_params)
                return (grad_d * uv[:, None, n_chan:]).sum(axis=2)

            if adaptive_step_size:
                Lu = 1
            else:
                Lu = compute_lipschitz(uv_hat, constants, 'u', b_hat_0)
            assert Lu > 0

            u_hat, pobj = fista(obj, grad_u, prox, 0.99 / Lu, u_hat, max_iter,
                                verbose=verbose, momentum=momentum, eps=eps,
                                adaptive_step_size=adaptive_step_size,
                                debug=debug, name="Update u")
            uv_hat = np.c_[u_hat, v_hat]

            # ---------------- update v
            def obj(v):
                uv = np.c_[u_hat, v]
                return objective(uv)

            def grad_v(v):
                uv = np.c_[u_hat, v]
                grad_d = gradient_d(uv, X=X, Z=Z, constants=constants,
                                    loss=loss, loss_params=loss_params)
                return (grad_d * uv[:, :n_chan, None]).sum(axis=1)

            if adaptive_step_size:
                Lv = 1
            else:
                Lv = compute_lipschitz(uv_hat, constants, 'v', b_hat_0)
            assert Lv > 0

            v_hat, pobj_v = fista(obj, grad_v, prox, 0.99 / Lv, v_hat,
                                  max_iter, momentum=momentum, eps=eps,
                                  adaptive_step_size=adaptive_step_size,
                                  verbose=verbose, debug=debug,
                                  name="Update v")
            uv_hat = np.c_[u_hat, v_hat]
            if debug:
                pobj.extend(pobj_v)

    elif solver_d == 'lbfgs':
        # use L-BFGS on joint [u, v] with a box constraint (L_inf norm <= 1)

        def func(uv):
            uv = np.reshape(uv, uv_hat0.shape)
            return objective(uv)

        def grad(uv):
            return gradient_uv(uv, constants=constants, flatten=True)

        def callback(uv):
            import matplotlib.pyplot as plt
            uv = np.reshape(uv, uv_hat0.shape)
            plt.figure('lbfgs')
            ax = plt.gca()
            if ax.lines == []:
                plt.plot(uv[:, n_chan:].T)
            else:
                for line, this in zip(ax.lines, uv[:, n_chan:]):
                    line.set_ydata(this)
            ax.relim()  # make sure all the data fits
            ax.autoscale_view(True, True, True)
            plt.draw()
            plt.pause(0.001)

        bounds = [(-1, 1) for idx in range(0, uv_hat0.size)]
        if debug:
            assert optimize.check_grad(func, grad, uv_hat0.ravel()) < 1e-5
            pobj = [objective(uv_hat0)]
        uv_hat, _, _ = optimize.fmin_l_bfgs_b(func, x0=uv_hat0.ravel(),
                                              fprime=grad, bounds=bounds,
                                              factr=1e7, callback=callback)
        uv_hat = np.reshape(uv_hat, uv_hat0.shape)
        if debug:
            pobj.append(objective(uv_hat))

    else:
        raise ValueError('Unknown solver_d: %s' % (solver_d, ))

    if debug:
        return uv_hat, pobj
    return uv_hat


def update_d(X, Z, D_hat0, b_hat_0=None, debug=False, max_iter=300, eps=None,
             solver_d='alternate', momentum=False, uv_constraint='separate',
             loss='l2', loss_params=dict(), verbose=0):
    """Learn d's in time domain.

    Parameters
    ----------
    X : array, shape (n_trials, n_times)
        The data for sparse coding
    Z : array, shape (n_atoms, n_trials, n_times - n_times_atom + 1)
        Can also be a list of n_trials LIL-sparse matrix of shape
            (n_atoms, n_times - n_times_atom + 1)
        The code for which to learn the atoms
    D_hat0 : array, shape (n_atoms, n_channels, n_times_atom)
        The initial atoms.
    b_hat_0 : array, shape (n_atoms * (n_channels + n_times_atom))
        Init eigen-vector vector used in power_iteration, used in warm start.
    debug : bool
        If True, return the cost at each iteration.
    momentum : bool
        If True, use an accelerated version of the proximal gradient descent.
    solver_d : str in {'fista', 'lbfgs'}
        The type of solver to update d:
        If 'fista', the solver optimize D with fista and line search
        If 'lbfgs', the solver uses lbfgs with box constraints
    loss : str in {'l2' | 'dtw' | 'whitening'}
        The data-fit
    loss_params : dict
        Parameters of the loss
    verbose : int
        Verbosity level.

    Returns
    -------
    D_hat : array, shape (n_atoms, n_channels, n_times_atom)
        The atoms to learn from the data.
    """
    n_atoms, n_trials, n_times_valid = get_Z_shape(Z)
    _, n_chan, n_times = X.shape

    if loss == 'l2':
        constants = _get_d_update_constants(X, Z)
    else:
        constants = None

    def objective(D, full=False):
        if loss == 'l2':
            return compute_objective(D=D, constants=constants)
        return compute_X_and_objective_multi(X, Z, D_hat=D, loss=loss,
                                             loss_params=loss_params)

    if solver_d == 'fista':
        # use FISTA on joint [u, v], with an adaptive step size

        def grad(D):
            return gradient_d(D=D, X=X, Z=Z, constants=constants, loss=loss,
                              loss_params=loss_params)

        def prox(D):
            return prox_d(D)

        D_hat = fista(objective, grad, prox, None, D_hat0, max_iter,
                      verbose=verbose, momentum=momentum, eps=eps,
                      adaptive_step_size=True, debug=debug, name="Update D")

    elif solver_d == 'lbfgs':
        # use L-BFGS on joint [u, v] with a box constraint (L_inf norm <= 1)

        def func(D):
            D = np.reshape(D, D_hat0.shape)
            return objective(D)

        def grad(D):
            return gradient_d(D, constants=constants, flatten=True)

        bounds = [(-1, 1) for idx in range(0, D_hat0.size)]
        if debug:
            assert optimize.check_grad(func, grad, D_hat0.ravel()) < 1e-5
            pobj = [objective(D_hat0)]
        D_hat, _, _ = optimize.fmin_l_bfgs_b(func, x0=D_hat0.ravel(),
                                             fprime=grad, bounds=bounds,
                                             factr=1e7)
        D_hat = np.reshape(D_hat, D_hat0.shape)
        if debug:
            pobj.append(objective(D_hat))

    else:
        raise ValueError('Unknown solver_d: %s' % (solver_d, ))

    if debug:
        return D_hat, pobj
    return D_hat


def _get_d_update_constants(X, Z):
    n_atoms, n_trials, n_times_valid = get_Z_shape(Z)
    n_trials, n_chan, n_times = X.shape
    n_times_atom = n_times - n_times_valid + 1

    if is_list_of_lil(Z):
        ZtX = _fast_compute_ZtX(Z, X)
        ZtZ = _fast_compute_ZtZ(Z, n_times_atom)
    else:
        ZtX = compute_ZtX(Z, X)
        ZtZ = compute_ZtZ(Z, n_times_atom)

    constants = {}
    constants['ZtX'] = ZtX
    constants['ZtZ'] = ZtZ
    constants['n_chan'] = X.shape[1]
    constants['XtX'] = np.sum(X * X)
    return constants


def compute_lipschitz(uv0, constants, variable, b_hat_0=None):

    n_chan = constants['n_chan']
    u0, v0 = uv0[:, :n_chan], uv0[:, n_chan:]
    n_atoms = uv0.shape[0]
    n_times_atom = uv0.shape[1] - n_chan
    if b_hat_0 is None:
        b_hat_0 = np.random.randn(uv0.size)

    def op_Hu(u):
        u = np.reshape(u, (n_atoms, n_chan))
        uv = np.c_[u, v0]
        H_d = numpy_convolve_uv(constants['ZtZ'], uv)
        H_u = (H_d * uv[:, None, n_chan:]).sum(axis=2)
        return H_u.ravel()

    def op_Hv(v):
        v = np.reshape(v, (n_atoms, n_times_atom))
        uv = np.c_[u0, v]
        H_d = numpy_convolve_uv(constants['ZtZ'], uv)
        H_v = (H_d * uv[:, :n_chan, None]).sum(axis=1)
        return H_v.ravel()

    if variable == 'u':
        b_hat_u0 = b_hat_0.reshape(n_atoms, -1)[:, :n_chan].ravel()
        n_points = n_atoms * n_chan
        L = power_iteration(op_Hu, n_points, b_hat_0=b_hat_u0)
    elif variable == 'v':
        b_hat_v0 = b_hat_0.reshape(n_atoms, -1)[:, n_chan:].ravel()
        n_points = n_atoms * n_times_atom
        L = power_iteration(op_Hv, n_points, b_hat_0=b_hat_v0)
    return L
