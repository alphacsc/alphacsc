# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>

import numpy as np

from . import cython_code
from .utils.lil import get_z_shape, is_list_of_lil
from .utils.optim import fista, power_iteration
from .utils.convolution import numpy_convolve_uv
from .utils.compute_constants import compute_ztz, compute_ztX
from .utils.dictionary import tukey_window

from .loss_and_gradient import compute_objective, compute_X_and_objective_multi
from .loss_and_gradient import gradient_uv, gradient_d


def prox_uv(uv, uv_constraint='joint', n_channels=None, return_norm=False):
    if uv_constraint == 'joint':
        norm_uv = np.maximum(1, np.linalg.norm(uv, axis=1))
        uv /= norm_uv[:, None]

    elif uv_constraint == 'separate':
        assert n_channels is not None
        norm_u = np.maximum(1, np.linalg.norm(uv[:, :n_channels], axis=1))
        norm_v = np.maximum(1, np.linalg.norm(uv[:, n_channels:], axis=1))

        uv[:, :n_channels] /= norm_u[:, None]
        uv[:, n_channels:] /= norm_v[:, None]
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


def update_uv(X, z, uv_hat0, constants=None, b_hat_0=None, debug=False,
              max_iter=300, eps=None, solver_d='alternate', momentum=False,
              uv_constraint='separate', loss='l2', loss_params=dict(),
              verbose=0, window=False):
    """Learn d's in time domain.

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data for sparse coding
    z : array, shape (n_trials, n_atoms, n_times - n_times_atom + 1)
        Can also be a list of n_trials LIL-sparse matrix of shape
            (n_atoms, n_times - n_times_atom + 1)
        The code for which to learn the atoms
    uv_hat0 : array, shape (n_atoms, n_channels + n_times_atom)
        The initial atoms.
    constants : dict or None
        Dictionary of constants to accelerate the computation of the gradients.
        It should only be given for loss='l2' and should contain ztz and ztX.
    b_hat_0 : array, shape (n_atoms * (n_channels + n_times_atom))
        Init eigen-vector vector used in power_iteration, used in warm start.
    debug : bool
        If True, return the cost at each iteration.
    momentum : bool
        If True, use an accelerated version of the proximal gradient descent.
    uv_constraint : str in {'joint', 'separate'}
        The kind of norm constraint on the atoms:
        If 'joint', the constraint is norm_2([u, v]) <= 1
        If 'separate', the constraint is norm_2(u) <= 1 and norm_2(v) <= 1
    solver_d : str in {'alternate', 'joint'}
        The type of solver to update d:
        If 'alternate', the solver alternates between u then v
        If 'joint', the solver jointly optimize uv with a line search
    loss : str in {'l2' | 'dtw' | 'whitening'}
        The data-fit
    loss_params : dict
        Parameters of the loss
    verbose : int
        Verbosity level.
    window : boolean
        If True, reparametrize the atoms with a temporal Tukey window.

    Returns
    -------
    uv_hat : array, shape (n_atoms, n_channels + n_times_atom)
        The atoms to learn from the data.
    """
    n_trials, n_atoms, n_times_valid = get_z_shape(z)
    _, n_channels, n_times = X.shape
    n_times_atom = uv_hat0.shape[1] - n_channels

    if window:
        tukey_window_ = tukey_window(n_times_atom)[None, :]
        uv_hat0 = uv_hat0.copy()
        uv_hat0[:, n_channels:] /= tukey_window_

    if solver_d == 'alternate':
        msg = "alternate solver should be used with separate constraints"
        assert uv_constraint == 'separate', msg

    if loss == 'l2' and constants is None:
        constants = _get_d_update_constants(X, z)

    def objective(uv):
        if window:
            uv = uv.copy()
            uv[:, n_channels:] *= tukey_window_
        if loss == 'l2':
            return compute_objective(D=uv, constants=constants)
        return compute_X_and_objective_multi(X, z, D_hat=uv, loss=loss,
                                             loss_params=loss_params,
                                             feasible_evaluation=True,
                                             uv_constraint=uv_constraint)

    if solver_d in ['joint', 'fista']:
        # use FISTA on joint [u, v], with an adaptive step size

        def grad(uv):
            if window:
                uv = uv.copy()
                uv[:, n_channels:] *= tukey_window_
            grad = gradient_uv(uv=uv, X=X, z=z, constants=constants, loss=loss,
                               loss_params=loss_params)
            if window:
                grad[:, n_channels:] *= tukey_window_
            return grad

        def prox(uv):
            if window:
                uv[:, n_channels:] *= tukey_window_
            uv = prox_uv(uv, uv_constraint=uv_constraint,
                         n_channels=n_channels)
            if window:
                uv[:, n_channels:] /= tukey_window_
            return uv

        uv_hat, pobj = fista(objective, grad, prox, None, uv_hat0, max_iter,
                             verbose=verbose, momentum=momentum, eps=eps,
                             adaptive_step_size=True, debug=debug,
                             name="Update uv")

    elif solver_d in ['alternate', 'alternate_adaptive']:
        # use FISTA on alternate u and v

        adaptive_step_size = (solver_d == 'alternate_adaptive')

        uv_hat = uv_hat0.copy()
        u_hat, v_hat = uv_hat[:, :n_channels], uv_hat[:, n_channels:]

        def prox_u(u):
            u /= np.maximum(1., np.linalg.norm(u, axis=1))[:, None]
            return u

        def prox_v(v):
            if window:
                v *= tukey_window_
            v /= np.maximum(1., np.linalg.norm(v, axis=1))[:, None]
            if window:
                v /= tukey_window_
            return v

        pobj = []
        for jj in range(1):
            # ---------------- update u

            def obj(u):
                uv = np.c_[u, v_hat]
                return objective(uv)

            def grad_u(u):
                if window:
                    uv = np.c_[u, v_hat * tukey_window_]
                else:
                    uv = np.c_[u, v_hat]
                grad_d = gradient_d(uv, X=X, z=z, constants=constants,
                                    loss=loss, loss_params=loss_params)
                return (grad_d * uv[:, None, n_channels:]).sum(axis=2)

            if adaptive_step_size:
                Lu = 1
            else:
                Lu = compute_lipschitz(uv_hat, constants, 'u', b_hat_0)
            assert Lu > 0

            u_hat, pobj_u = fista(obj, grad_u, prox_u, 0.99 / Lu, u_hat,
                                  max_iter, momentum=momentum, eps=eps,
                                  adaptive_step_size=adaptive_step_size,
                                  verbose=verbose, debug=debug,
                                  name="Update u")
            uv_hat = np.c_[u_hat, v_hat]
            if debug:
                pobj.extend(pobj_u)

            # ---------------- update v
            def obj(v):
                uv = np.c_[u_hat, v]
                return objective(uv)

            def grad_v(v):
                if window:
                    v = v * tukey_window_
                uv = np.c_[u_hat, v]
                grad_d = gradient_d(uv, X=X, z=z, constants=constants,
                                    loss=loss, loss_params=loss_params)
                grad_v = (grad_d * uv[:, :n_channels, None]).sum(axis=1)
                if window:
                    grad_v *= tukey_window_
                return grad_v

            if adaptive_step_size:
                Lv = 1
            else:
                Lv = compute_lipschitz(uv_hat, constants, 'v', b_hat_0)
            assert Lv > 0

            v_hat, pobj_v = fista(obj, grad_v, prox_v, 0.99 / Lv, v_hat,
                                  max_iter, momentum=momentum, eps=eps,
                                  adaptive_step_size=adaptive_step_size,
                                  verbose=verbose, debug=debug,
                                  name="Update v")
            uv_hat = np.c_[u_hat, v_hat]
            if debug:
                pobj.extend(pobj_v)

    else:
        raise ValueError('Unknown solver_d: %s' % (solver_d, ))

    if window:
        uv_hat[:, n_channels:] *= tukey_window_

    if debug:
        return uv_hat, pobj
    return uv_hat


def update_d(X, z, D_hat0, constants=None, b_hat_0=None, debug=False,
             max_iter=300, eps=None, solver_d='fista', momentum=False,
             uv_constraint='joint', loss='l2', loss_params=dict(), verbose=0,
             window=False):
    """Learn d's in time domain.

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data for sparse coding
    z : array, shape (n_trials, n_atoms, n_times - n_times_atom + 1)
        Can also be a list of n_trials LIL-sparse matrix of shape
            (n_atoms, n_times - n_times_atom + 1)
        The code for which to learn the atoms
    D_hat0 : array, shape (n_atoms, n_channels, n_times_atom)
        The initial atoms.
    constants : dict or None
        Dictionary of constants to accelerate the computation of the gradients.
        It should only be given for loss='l2' and should contain ztz and ztX.
    b_hat_0 : array, shape (n_atoms * (n_channels + n_times_atom))
        Init eigen-vector vector used in power_iteration, used in warm start.
    debug : bool
        If True, return the cost at each iteration.
    momentum : bool
        If True, use an accelerated version of the proximal gradient descent.
    solver_d : str in {'fista'}
        The type of solver to update d:
        If 'fista', the solver optimize D with fista and line search
    loss : str in {'l2' | 'dtw' | 'whitening'}
        The data-fit
    loss_params : dict
        Parameters of the loss
    verbose : int
        Verbosity level.
    window : boolean
        If True, reparametrize the atoms with a temporal Tukey window.

    Returns
    -------
    D_hat : array, shape (n_atoms, n_channels, n_times_atom)
        The atoms to learn from the data.
    """
    n_trials, n_atoms, n_times_valid = get_z_shape(z)
    _, n_channels, n_times = X.shape
    n_atoms, n_channels, n_times_atom = D_hat0.shape

    if window:
        tukey_window_ = tukey_window(n_times_atom)[None, None, :]
        D_hat0 = D_hat0 / tukey_window_

    if loss == 'l2' and constants is None:
        constants = _get_d_update_constants(X, z)

    def objective(D, full=False):
        if window:
            D = D * tukey_window_
        if loss == 'l2':
            return compute_objective(D=D, constants=constants)
        return compute_X_and_objective_multi(X, z, D_hat=D, loss=loss,
                                             loss_params=loss_params)

    if True:  # only solver available here
        # use FISTA on joint [u, v], with an adaptive step size

        def grad(D):
            if window:
                D = D * tukey_window_
            grad = gradient_d(D=D, X=X, z=z, constants=constants, loss=loss,
                              loss_params=loss_params)
            if window:
                grad *= tukey_window_
            return grad

        def prox(D):
            if window:
                D *= tukey_window_
            D = prox_d(D)
            if window:
                D /= tukey_window_
            return D

        D_hat, pobj = fista(objective, grad, prox, None, D_hat0, max_iter,
                            verbose=verbose, momentum=momentum, eps=eps,
                            adaptive_step_size=True, debug=debug,
                            name="Update D")

    else:
        raise ValueError('Unknown solver_d: %s' % (solver_d, ))

    if window:
        D_hat = D_hat * tukey_window_

    if debug:
        return D_hat, pobj
    return D_hat


def _get_d_update_constants(X, z):
    n_trials, n_atoms, n_times_valid = get_z_shape(z)
    n_trials, n_channels, n_times = X.shape
    n_times_atom = n_times - n_times_valid + 1

    if is_list_of_lil(z):
        cython_code._assert_cython()
        ztX = cython_code._fast_compute_ztX(z, X)
        ztz = cython_code._fast_compute_ztz(z, n_times_atom)
    else:
        ztX = compute_ztX(z, X)
        ztz = compute_ztz(z, n_times_atom)

    constants = {}
    constants['ztX'] = ztX
    constants['ztz'] = ztz
    constants['n_channels'] = X.shape[1]
    constants['XtX'] = np.dot(X.ravel(), X.ravel())
    return constants


def compute_lipschitz(uv0, constants, variable, b_hat_0=None):

    n_channels = constants['n_channels']
    u0, v0 = uv0[:, :n_channels], uv0[:, n_channels:]
    n_atoms = uv0.shape[0]
    n_times_atom = uv0.shape[1] - n_channels
    if b_hat_0 is None:
        b_hat_0 = np.random.randn(uv0.size)

    def op_Hu(u):
        u = np.reshape(u, (n_atoms, n_channels))
        uv = np.c_[u, v0]
        H_d = numpy_convolve_uv(constants['ztz'], uv)
        H_u = (H_d * uv[:, None, n_channels:]).sum(axis=2)
        return H_u.ravel()

    def op_Hv(v):
        v = np.reshape(v, (n_atoms, n_times_atom))
        uv = np.c_[u0, v]
        H_d = numpy_convolve_uv(constants['ztz'], uv)
        H_v = (H_d * uv[:, :n_channels, None]).sum(axis=1)
        return H_v.ravel()

    if variable == 'u':
        b_hat_u0 = b_hat_0.reshape(n_atoms, -1)[:, :n_channels].ravel()
        n_points = n_atoms * n_channels
        L = power_iteration(op_Hu, n_points, b_hat_0=b_hat_u0)
    elif variable == 'v':
        b_hat_v0 = b_hat_0.reshape(n_atoms, -1)[:, n_channels:].ravel()
        n_points = n_atoms * n_times_atom
        L = power_iteration(op_Hv, n_points, b_hat_0=b_hat_v0)
    return L
