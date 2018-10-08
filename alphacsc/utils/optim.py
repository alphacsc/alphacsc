import time

import numpy as np
from scipy import optimize

from .compute_constants import compute_DtD
from . import check_random_state


def _support_least_square(X, uv, z, debug=False):
    """WIP, not fonctional!"""
    n_trials, n_channels, n_times = X.shape
    n_atoms, _, n_times_valid = z.shape
    n_times_atom = n_times - n_times_valid + 1

    # Compute DtD
    DtD = compute_DtD(uv, n_channels)
    t0 = n_times_atom - 1
    z_hat = np.zeros(z.shape)

    for idx in range(n_trials):
        Xi = X[idx]
        support_i = z[:, idx].nonzero()
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
            aux_i = np.dot(uv[k_i, :n_channels], Xi[:, t_i:t_i + n_times_atom])
            lhs[i] = np.dot(uv[k_i, n_channels:], aux_i)

        # Solve the non-negative least-square with nnls
        z_star, a = optimize.nnls(rhs, lhs)
        for i, (k_i, t_i) in enumerate(zip(*support_i)):
            z_hat[k_i, idx, t_i] = z_star[i]

    return z_hat


def fista(f_obj, f_grad, f_prox, step_size, x0, max_iter, verbose=0,
          momentum=False, eps=None, adaptive_step_size=False, debug=False,
          scipy_line_search=True, name='ISTA', timing=False, restart=None):
    """ISTA and FISTA algorithm

    Parameters
    ----------
    f_obj : callable
        Objective function. Used only if debug or adaptive_step_size.
    f_grad : callable
        Gradient of the objective function
    f_prox : callable
        Proximal operator
    step_size : float or None
        Step size of each update. Can be None if adaptive_step_size.
    x0 : array
        Initial point of the optimization
    max_iter : int
        Maximum number of iterations
    verbose : int
        Verbosity level
    momentum : boolean
        If True, use FISTA instead of ISTA
    eps : float or None
        Tolerance for the stopping criterion
    adaptive_step_size : boolean
        If True, the step size is adapted at each step
    debug : boolean
        If True, compute the objective function at each step and return the
        list at the end.
    timing : boolean
        If True, compute the objective function at each step, and the duration
        of each step, and return both lists at the end.
    restart : int or None
        If not None, restart the momentum every `restart` iterations.

    Returns
    -------
    x_hat : array
        The final point after optimization
    pobj : list or None
        If debug is True, pobj contains the value of the cost function at each
        iteration.
    """
    pobj = None
    if debug:
        pobj = list()
    if timing:
        times = [0]
        pobj = [f_obj(x0)]
        start = time.time()

    if step_size is None:
        step_size = 1.
    if eps is None:
        eps = np.finfo(np.float32).eps
    obj_uv = None

    tk = 1.0
    x_hat = x0.copy()
    x_hat_aux = x_hat.copy()
    grad = np.empty(x_hat.shape)
    diff = np.empty(x_hat.shape)
    for ii in range(max_iter):
        # restart every n iterations
        if restart is not None and ii > 0 and (ii % restart) == 0:
            x_hat_aux = x_hat.copy()
            tk = 1.0

        grad[:] = f_grad(x_hat_aux)

        if adaptive_step_size:
            if scipy_line_search:

                def f_obj_(x_hat):
                    x_hat = np.reshape(x_hat, x0.shape)
                    return f_obj(f_prox(x_hat))

                step_size, _, obj_uv = optimize.linesearch.line_search_armijo(
                    f_obj_, x_hat.ravel(), -grad.ravel(), grad.ravel(), obj_uv,
                    c1=1e-5, alpha0=step_size)
                if step_size is None:
                    step_size = 0
                x_hat_aux -= step_size * grad
                x_hat_aux = f_prox(x_hat_aux)

            else:

                def f(step_size):
                    x_hat = f_prox(x_hat_aux - step_size * grad)
                    pobj = f_obj(x_hat)
                    return pobj, x_hat

                obj_uv, x_hat_aux, step_size = _adaptive_step_size(
                    f, obj_uv, alpha=step_size)

        else:
            x_hat_aux -= step_size * grad
            x_hat_aux = f_prox(x_hat_aux)

        diff[:] = x_hat_aux - x_hat
        x_hat[:] = x_hat_aux
        if momentum:
            tk_new = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
            x_hat_aux += (tk - 1) / tk_new * diff
            tk = tk_new

        if debug:
            pobj.append(f_obj(x_hat))
        if timing:
            times.append(time.time() - start)
            pobj.append(f_obj(x_hat))
            start = time.time()

        f = np.sum(abs(diff))
        if f <= eps:
            break
        if f > 1e50:
            raise RuntimeError("The D update have diverged.")
    else:
        if verbose > 1:
            print('[{}] update did not converge'.format(name))
    if verbose > 5:
        print('[{}]: {} iterations'.format(name, ii + 1))

    if timing:
        return x_hat, pobj, times
    return x_hat, pobj


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


def power_iteration(lin_op, n_points=None, b_hat_0=None, max_iter=1000,
                    tol=1e-7, random_state=None):
    """Estimate dominant eigenvalue of linear operator A.

    Parameters
    ----------
    lin_op : callable or array
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
    if hasattr(lin_op, 'dot'):
        n_points = lin_op.shape[1]
        lin_op = lin_op.dot
    elif callable(lin_op):
        msg = ("power_iteration require n_points argument when lin_op is "
               "callable")
        assert n_points is not None, msg
    else:
        raise ValueError("lin_op should be a callable or a ndarray")

    rng = check_random_state(random_state)
    if b_hat_0 is None:
        b_hat = rng.rand(n_points)
    else:
        b_hat = b_hat_0

    mu_hat = np.nan
    for ii in range(max_iter):
        b_hat = lin_op(b_hat)
        norm = np.linalg.norm(b_hat)
        if norm == 0:
            return 0
        b_hat /= norm
        fb_hat = lin_op(b_hat)
        mu_old = mu_hat
        mu_hat = np.dot(b_hat, fb_hat)
        # note, we might exit the loop before b_hat converges
        # since we care only about mu_hat converging
        if (mu_hat - mu_old) / mu_old < tol:
            break

    assert not np.isnan(mu_hat)

    if b_hat_0 is not None:
        # copy inplace into b_hat_0 for next call to power_iteration
        np.copyto(b_hat_0, b_hat)

    return mu_hat
