import numpy as np
from scipy import optimize

from .compute_constants import _compute_DtD


def _support_least_square(X, uv, Z, debug=False):
    """WIP, not fonctional!"""
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
        for i, (k_i, t_i) in enumerate(zip(*support_i)):
            Z_hat[k_i, idx, t_i] = z_star[i]

    return Z_hat


def fista(f_obj, f_grad, f_prox, step_size, x0, max_iter, verbose=0,
          momentum=False, eps=None, adaptive_step_size=False, debug=False,
          scipy_line_search=True):
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
        If true, compute the objective function at each step and return the
        list at the end.

    Returns
    -------
    x_hat : array
        The final point after optimization
    """

    if debug:
        pobj = list()
    if step_size is None:
        step_size = 1.
    if eps is None:
        eps = np.finfo(np.float32).eps
    obj_uv = None

    tk = 1
    x_hat = x0.copy()
    x_hat_aux = x_hat.copy()
    grad = np.empty(x_hat.shape)
    diff = np.empty(x_hat.shape)
    for ii in range(max_iter):
        grad[:] = f_grad(x_hat_aux)

        if adaptive_step_size:
            if scipy_line_search:

                def f_obj_(x_hat):
                    x_hat = np.reshape(x_hat, x0.shape)
                    return f_obj(f_prox(x_hat))

                step_size, _, obj_uv = optimize.linesearch.line_search_armijo(
                    f_obj_, x_hat.ravel(), -grad.ravel(), grad.ravel(), obj_uv,
                    c1=1e-4, alpha0=step_size)
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
            pobj.append(f_obj(x_hat, full=True))
        f = np.sum(abs(diff))
        if f <= eps:
            break
        if f > 1e50:
            raise RuntimeError("The D update have diverged.")
    else:
        if verbose > 1:
            print('update [fista] did not converge')
    if verbose > 1:
        print('ISTA: %d iterations' % (ii + 1))

    if debug:
        return x_hat, pobj
    return x_hat
