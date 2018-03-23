"""Convolutional dictionary learning"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>

import numpy as np
from numpy import convolve
from numba import jit
from scipy import optimize

from .utils import check_random_state
from .utils.optim import fista
from .utils.convolution import numpy_convolve_uv

from .loss_and_gradient import compute_objective, compute_X_and_objective_multi
from .loss_and_gradient import gradient_uv, gradient_d


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


def update_uv(X, Z, uv_hat0, b_hat_0=None, debug=False, max_iter=300, eps=None,
              solver_d='alternate', momentum=False, uv_constraint='separate',
              loss='l2', loss_params=dict(), verbose=0):
    """Learn d's in time domain.

    Parameters
    ----------
    X : array, shape (n_trials, n_times)
        The data for sparse coding
    Z : array, shape (n_atoms, n_trials, n_times - n_times_atom + 1)
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
    loss : str in {'l2' | 'dtw'}
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
    n_atoms, n_trials, n_times_valid = Z.shape
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

    def objective(uv, full=False):
        if loss == 'l2':
            return compute_objective(uv=uv, constants=constants)
        return compute_X_and_objective_multi(X, Z, uv_hat=uv, loss=loss,
                                             loss_params=loss_params)

    if solver_d == 'joint':
        # use FISTA on joint [u, v], with an adaptive step size

        def grad(uv):
            return gradient_uv(uv=uv, X=X, Z=Z, constants=constants, loss=loss,
                               loss_params=loss_params)

        def prox(uv):
            return prox_uv(uv, uv_constraint=uv_constraint, n_chan=n_chan)

        uv_hat = fista(objective, grad, prox, None, uv_hat0, max_iter,
                       verbose=verbose, momentum=momentum, eps=eps,
                       adaptive_step_size=True, debug=debug)
        if debug:
            uv_hat, pobj = uv_hat

    elif solver_d in ['alternate', 'alternate_adaptive']:
        # use FISTA on alternate u and v

        adaptive_step_size = (solver_d == 'alternate_adaptive')

        pobj = list()
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
                grad_d = gradient_d(X=X, Z=Z, uv=uv, constants=constants,
                                    loss=loss, loss_params=loss_params)
                return (grad_d * uv[:, None, n_chan:]).sum(axis=2)

            if adaptive_step_size:
                Lu = 1
            else:
                Lu = compute_lipschitz(uv_hat, constants, 'u', b_hat_0)
            assert Lu > 0

            u_hat = fista(obj, grad_u, prox, 0.99 / Lu, u_hat, max_iter,
                          verbose=verbose, momentum=momentum, eps=eps,
                          adaptive_step_size=adaptive_step_size)
            uv_hat = np.c_[u_hat, v_hat]
            if debug:
                pobj.append(objective(uv_hat, full=True))

            # ---------------- update v
            def obj(v):
                uv = np.c_[u_hat, v]
                return objective(uv)

            def grad_v(v):
                uv = np.c_[u_hat, v]
                grad_d = gradient_d(uv=uv, X=X, Z=Z, constants=constants,
                                    loss=loss, loss_params=loss_params)
                return (grad_d * uv[:, :n_chan, None]).sum(axis=1)

            if adaptive_step_size:
                Lv = 1
            else:
                Lv = compute_lipschitz(uv_hat, constants, 'v', b_hat_0)
            assert Lv > 0

            v_hat = fista(obj, grad_v, prox, 0.99 / Lv, v_hat, max_iter,
                          verbose=verbose, momentum=momentum, eps=eps,
                          adaptive_step_size=adaptive_step_size)
            uv_hat = np.c_[u_hat, v_hat]
            if debug:
                pobj.append(objective(uv_hat, full=True))

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


def _get_d_update_constants(X, Z):
    # Get shapes
    n_atoms, n_trials, n_times_valid = Z.shape
    _, n_chan, n_times = X.shape
    n_times_atom = n_times - n_times_valid + 1

    ZtX = np.zeros((n_atoms, n_chan, n_times_atom))
    for k, n, t in zip(*Z.nonzero()):
        ZtX[k, :, :] += Z[k, n, t] * X[n, :, t:t + n_times_atom]

    constants = {}
    constants['ZtX'] = ZtX

    assert np.allclose(ZtX, constants['ZtX'])

    ZtZ = compute_ZtZ(Z, n_times_atom)
    constants['ZtZ'] = ZtZ
    constants['n_chan'] = n_chan
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


@jit()
def compute_ZtZ(Z, n_times_atom):
    """
    ZtZ.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
    Z.shape = n_atoms, n_trials, n_times - n_times_atom + 1)
    """
    # TODO: benchmark the cross correlate function of numpy
    n_atoms, n_trials, n_times_valid = Z.shape

    ZtZ = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    t0 = n_times_atom - 1
    for k0 in range(n_atoms):
        for k in range(n_atoms):
            for i in range(n_trials):
                for t in range(n_times_atom):
                    if t == 0:
                        ZtZ[k0, k, t0] += (Z[k0, i] * Z[k, i]).sum()
                    else:
                        ZtZ[k0, k, t0 + t] += (
                            Z[k0, i, :-t] * Z[k, i, t:]).sum()
                        ZtZ[k0, k, t0 - t] += (
                            Z[k0, i, t:] * Z[k, i, :-t]).sum()
    return ZtZ


def power_iteration(lin_op, n_points, b_hat_0=None, max_iter=1000, tol=1e-7,
                    random_state=None):
    """Estimate dominant eigenvalue of linear operator A.

    Parameters
    ----------
    lin_op : callable
        Linear operator from which we estimate the largest eigenvalue.
    n_points : tuple
        Input shape of the linear operator `lin_op`.
    b_hat_0 : array, shape (n_points, )
        Init vector. The estimated eigen-vector is stored inplace in `b_hat_0`
        to allow warm start of future call of this function with the same
        variable.

    Returns
    -------
    mu_hat : float
        The largest eigenvalue
    """
    rng = check_random_state(random_state)
    if b_hat_0 is None:
        b_hat = rng.rand(n_points)
    else:
        b_hat = b_hat_0

    mu_hat = np.nan
    for ii in range(max_iter):
        b_hat = lin_op(b_hat)
        b_hat /= np.linalg.norm(b_hat)
        fb_hat = lin_op(b_hat)
        mu_old = mu_hat
        mu_hat = np.dot(b_hat, fb_hat)
        # note, we might exit the loop before b_hat converges
        # since we care only about mu_hat converging
        if (mu_hat - mu_old) / mu_old < tol:
            break

    if b_hat_0 is not None:
        # copy inplace into b_hat_0 for next call to power_iteration
        np.copyto(b_hat_0, b_hat)

    return mu_hat


def _adaptive_step_size(f, f0=None, alpha=None, tau=2):
    """
    Parameters
    ----------
    f : callable
        Optimized function, take only the step size as argument
    f0 : float
        value of f at current point, i.e. step size = 0
    alpha : float
        Initial step size
    tau : float
        Multiplication factor of the step size during the adaptation
    """

    if alpha is None:
        alpha = 1

    if f0 is None:
        f0, _ = f(0)
    f_alpha, x_alpha = f(alpha)
    f_alpha_down, x_alpha_down = f(alpha / tau)
    f_alpha_up, x_alpha_up = f(alpha * tau)

    alphas = [0, alpha / tau, alpha, alpha * tau]
    fs = [f0, f_alpha_down, f_alpha, f_alpha_up]
    xs = [None, x_alpha_down, x_alpha, x_alpha_up]
    i = np.argmin(fs)
    if i == 0:
        alpha /= tau * tau
        f_alpha, x_alpha = f(alpha)
        while f0 <= f_alpha and alpha > 1e-20:
            alpha /= tau
            f_alpha, x_alpha = f(alpha)
        return f_alpha, x_alpha, alpha
    else:
        return fs[i], xs[i], alphas[i]
