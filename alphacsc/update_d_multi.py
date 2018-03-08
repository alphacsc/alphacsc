"""Convolutional dictionary learning"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>

import numpy as np
from numpy import convolve
from numba import jit
import functools
from scipy import optimize

from .utils import construct_X_multi, _get_D, check_random_state


PHI = (np.sqrt(5) + 1) / 2


def tensordot_convolve(ZtZ, D):
    """Compute the multivariate (valid) convolution of ZtZ and D

    Parameters
    ----------
    ZtZ: array, shape = (n_atoms, n_atoms, 2 * n_times_atom - 1)
        Activations
    D: array, shape = (n_atoms, n_channels, n_times_atom)
        Dictionnary

    Returns
    -------
    G : array, shape = (n_atoms, n_channels, n_times_atom)
        Gradient
    """
    n_atoms, n_channels, n_times_atom = D.shape
    D_revert = D[:, :, ::-1]

    G = np.zeros(D.shape)
    for t in range(n_times_atom):
        G[:, :, t] = np.tensordot(ZtZ[:, :, t:t + n_times_atom], D_revert,
                                  axes=([1, 2], [0, 2]))
    return G


def numpy_convolve_uv(ZtZ, uv):
    """Compute the multivariate (valid) convolution of ZtZ and D

    Parameters
    ----------
    ZtZ: array, shape = (n_atoms, n_atoms, 2 * n_times_atom - 1)
        Activations
    uv: array, shape = (n_atoms, n_channels + n_times_atom)
        Dictionnary

    Returns
    -------
    G : array, shape = (n_atoms, n_channels, n_times_atom)
        Gradient
    """
    assert uv.ndim == 2
    n_times_atom = (ZtZ.shape[2] + 1) // 2
    n_atoms = ZtZ.shape[0]
    n_channels = uv.shape[1] - n_times_atom

    u = uv[:, :n_channels]
    v = uv[:, n_channels:]

    G = np.zeros((n_atoms, n_channels, n_times_atom))
    for k0 in range(n_atoms):
        for k1 in range(n_atoms):
            G[k0, :, :] += (convolve(ZtZ[k0, k1], v[k1], mode='valid')[None, :]
                            * u[k1, :][:, None])

    return G


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


def _gradient_d(D, X=None, Z=None, constants=None, uv=None, n_chan=None):
    if constants:
        if D is None:
            assert uv is not None
            g = numpy_convolve_uv(constants['ZtZ'], uv)
        else:
            g = tensordot_convolve(constants['ZtZ'], D)
        return g - constants['ZtX']
    else:
        if D is None:
            assert uv is not None and n_chan is not None
            D = _get_D(uv, n_chan)
        residual = construct_X_multi(Z, D) - X
        return _dense_transpose_convolve(Z, residual)


def _gradient_uv(uv, X=None, Z=None, constants=None):
    if constants:
        n_chan = constants['n_chan']
    else:
        assert X is not None
        assert Z is not None
        n_chan = X.shape[1]
    grad_d = _gradient_d(None, X, Z, constants, uv=uv, n_chan=n_chan)
    grad_u = (grad_d * uv[:, None, n_chan:]).sum(axis=2)
    grad_v = (grad_d * uv[:, :n_chan, None]).sum(axis=1)
    return np.c_[grad_u, grad_v]


def _shifted_objective_uv(uv, constants):
    n_chan = constants['n_chan']
    grad_d = .5 * numpy_convolve_uv(constants['ZtZ'], uv) - constants['ZtX']
    cost = (grad_d * uv[:, None, n_chan:]).sum(axis=2)
    return np.dot(cost.ravel(), uv[:, :n_chan].ravel())


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


def fista(grad, prox, step_size, x0, max_iter, verbose=0,
          momentum=False, eps=None):

    if eps is None:
        eps = np.finfo(np.float32).eps
    tk = 1
    x_hat = x0.copy()
    z_hat = x_hat.copy()
    diff = np.empty(x_hat.shape)
    for ii in range(max_iter):
        z_hat -= step_size * grad(z_hat)
        prox(z_hat)
        diff[:] = z_hat - x_hat
        x_hat[:] = z_hat
        if momentum:
            tk_new = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
            z_hat += (tk - 1) / tk_new * diff
            tk = tk_new
        f = np.sum(abs(diff))
        if f <= eps:
            break
        if f > 1e50:
            raise RuntimeError("The D update have diverged.")
    else:
        if verbose > 1:
            print('update [fista] did not converge')
    if verbose > 1:
        print('%d iterations' % (ii + 1))

    return x_hat


def update_uv(X, Z, uv_hat0, b_hat_0=None, debug=False, max_iter=300, eps=None,
              solver_d='alternate', momentum=False, uv_constraint='separate',
              verbose=0):
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

    # XXX : FISTA does not work and the cost goes up, should be fixed.
    constants = _get_d_update_constants(X, Z, b_hat_0=b_hat_0)

    def objective(uv, full=False):
        cost = _shifted_objective_uv(uv, constants)
        if full:
            cost += np.sum(X * X) / 2
        return cost

    def gradient(uv):
        uv = uv.reshape((n_atoms, -1))
        return _gradient_uv(uv, constants=constants).ravel()

    if solver_d == 'joint':
        # TODO: add a line-search

        if debug:
            pobj = list()

        if eps is None:
            eps = np.finfo(np.float32).eps
        tk = 1
        uv_hat = uv_hat0.copy()
        uv_hat_aux = uv_hat.copy()
        grad = np.empty(uv_hat.shape)
        diff = np.empty(uv_hat.shape)
        alpha = None
        obj_uv = None
        for ii in range(max_iter):

            grad[:] = _gradient_uv(uv_hat_aux, constants=constants)

            def f(step_size):
                uv = prox_uv(uv_hat_aux - step_size * grad,
                             uv_constraint=uv_constraint, n_chan=n_chan)
                return objective(uv), uv

            obj_uv, uv_hat_aux, alpha = _adaptive_step_size(
                f, obj_uv, alpha=alpha)
            diff[:] = uv_hat_aux - uv_hat
            uv_hat[:] = uv_hat_aux
            if momentum:  # TODO: FISTA does not work well!
                tk_new = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
                uv_hat_aux += (tk - 1) / tk_new * diff
                tk = tk_new
            if debug:
                pobj.append(objective(uv_hat, full=True))
            f = np.sum(abs(diff))
            if f <= eps:
                break
            if f > 1e50:
                raise RuntimeError("The D update have diverged.")
        else:
            if verbose > 1:
                print('update_uv did not converge')
        if verbose > 1:
            print('%d iterations' % (ii + 1))

    elif solver_d == 'alternate':

        pobj = list()
        uv_hat = uv_hat0.copy()
        u_hat, v_hat = uv_hat[:, :n_chan], uv_hat[:, n_chan:]

        def prox(u):
            u /= np.maximum(1., np.linalg.norm(u, axis=1))[:, None]
            return u

        for jj in range(5):
            # update u
            def grad_u(u):
                uv = np.c_[u, v_hat]
                grad_d = _gradient_d(None, constants=constants, uv=uv,
                                     n_chan=n_chan)
                return (grad_d * uv[:, None, n_chan:]).sum(axis=2)

            Lu = compute_lipschitz(uv_hat, constants, 'u', b_hat_0)
            assert Lu > 0
            u_hat = fista(grad_u, prox, 0.99 / Lu, u_hat, max_iter)
            uv_hat = np.c_[u_hat, v_hat]
            if debug:
                pobj.append(objective(uv_hat, full=True))

            # update v
            def grad_v(v):
                uv = np.c_[u_hat, v]
                grad_d = _gradient_d(None, constants=constants, uv=uv,
                                     n_chan=n_chan)
                return (grad_d * uv[:, :n_chan, None]).sum(axis=1)
            Lv = compute_lipschitz(uv_hat, constants, 'v', b_hat_0)
            assert Lv > 0
            v_hat = fista(grad_v, prox, 0.99 / Lv, v_hat, max_iter)
            uv_hat = np.c_[u_hat, v_hat]
            if debug:
                pobj.append(objective(uv_hat, full=True))

    elif solver_d == 'lbfgs':
        constants = _get_d_update_constants(X, Z, b_hat_0=b_hat_0)

        def func(uv):
            uv = np.reshape(uv, uv_hat0.shape)
            return objective(uv)

        def grad(uv):
            uv = np.reshape(uv, uv_hat0.shape)
            return _gradient_uv(uv, constants=constants).ravel()

        def callback(uv):
            import matplotlib.pyplot as plt
            uv = np.reshape(uv, uv_hat0.shape)
            plt.figure(0)
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


def _get_d_update_constants(X, Z, b_hat_0=None):
    # Get shapes
    n_atoms, n_trials, n_times_valid = Z.shape
    _, n_chan, n_times = X.shape
    n_times_atom = n_times - n_times_valid + 1

    constants = {}
    constants['ZtX'] = np.sum(
        [[[convolve(zik[::-1], xip, mode='valid') for xip in xi]
          for zik, xi in zip(zk, X)] for zk in Z], axis=1)

    ZtZ = compute_ZtZ(Z, n_times_atom)
    constants['ZtZ'] = ZtZ
    constants['n_chan'] = n_chan
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
