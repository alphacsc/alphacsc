# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>

import numpy as np
from scipy import linalg, optimize

from .utils import construct_X, check_consistent_shape


def update_d(X, Z, n_times_atom, lambd0=None, ds_init=None, debug=False,
             solver_kwargs=dict(), sample_weights=None, verbose=0):
    """Learn d's in time domain.

    Parameters
    ----------
    X : array, shape (n_trials, n_times)
        The data for sparse coding
    Z : array, shape (n_atoms, n_trials, n_times - n_times_atom + 1)
        The code for which to learn the atoms
    n_times_atom : int
        The shape of atoms.
    lambd0 : array, shape (n_atoms,) | None
        The init for lambda.
    debug : bool
        If True, check grad.
    solver_kwargs : dict
        Parameters for the solver
    sample_weights: array, shape (n_trials, n_times)
        Weights applied on the cost function.
    verbose : int
        Verbosity level.

    Returns
    -------
    d_hat : array, shape (k, n_times_atom)
        The atom to learn from the data.
    lambd_hats : float
        The dual variables
    """
    check_consistent_shape(X, sample_weights)
    n_atoms, n_trials, n_times_valid = Z.shape
    n_times = n_times_valid + n_times_atom - 1

    if lambd0 is None:
        lambd0 = 10. * np.ones(n_atoms)

    lhs = np.zeros((n_times_atom * n_atoms, ) * 2)
    rhs = np.zeros(n_times_atom * n_atoms)
    Zki = np.zeros(n_times + (n_times_atom - 1))
    for i in range(n_trials):

        ZZi = []
        for k in range(n_atoms):
            Zki[n_times_atom - 1:-(n_times_atom - 1)] = Z[k, i]
            ZZik = _embed(Zki, n_times_atom)
            # n_times_atom, n_times = ZZik.shape
            ZZi.append(ZZik)

        ZZi = np.concatenate(ZZi, axis=0)
        # n_times_atom * n_atoms, n_times = ZZi.shape
        if sample_weights is not None:
            wZZi = sample_weights[i] * ZZi
        else:
            wZZi = ZZi
        lhs += np.dot(wZZi, ZZi.T)
        rhs += np.dot(wZZi, X[i])

    factr = solver_kwargs.get('factr', 1e7)  # default value
    d_hat, lambd_hats = solve_unit_norm_dual(lhs, rhs, lambd0=lambd0,
                                             factr=factr, debug=debug,
                                             lhs_is_toeplitz=False)
    d_hat = d_hat.reshape(n_atoms, n_times_atom)[:, ::-1]
    return d_hat, lambd_hats


def update_d_block(X, Z, n_times_atom, lambd0=None, ds_init=None,
                   projection='dual', n_iter=1, solver_kwargs=dict(),
                   sample_weights=None, verbose=0):
    """Learn d's in time domain but each atom is learned separately.

    Parameters
    ----------
    X : array, shape (n_trials, n_times)
        The data for sparse coding
    Z : array, shape (n_atoms, n_trials, n_times - n_times_atom + 1)
        The code from which to learn the atoms
    n_times_atom : int
        The size of atoms.
    lambd0 : array, shape (n_atoms,) | None
        The init for dual variables
    ds_init : array, shape (n_atoms, n_times_atom)
        Warm start d_hats for the l-bfgs / projected gradient solver.
    projection : str, ('primal', 'dual')
        Whether to project to unit ball in the primal or the dual.
    n_iter : int
        Number of outer loop over atoms
    solver_kwargs : dict
        Parameters for the solver
    sample_weights: array, shape (n_trials, n_times)
        Weights applied on the cost function.
    verbose : int
        Verbosity level.

    Returns
    -------
    d_hats : array, shape (n_atoms, n_times_atom)
        The atom to learn from the data.
    lambd_hats : float
        The dual variables
    """
    check_consistent_shape(X, sample_weights)
    n_atoms, n_trials, n_times_valid = Z.shape
    n_times = n_times_valid + n_times_atom - 1

    if lambd0 is None:
        lambd0 = 10. * np.ones(n_atoms)

    Zki = np.zeros(n_times + n_times_atom - 1)

    if ds_init is None:
        ds = np.zeros((n_atoms, n_times_atom))
    else:
        ds = ds_init.copy()

    lambd_hats = np.zeros(n_atoms)
    residual = X - construct_X(Z, ds)
    if verbose > 1:
        print('Using method %s for projection' % projection)
    for _ in range(n_iter):
        for k in range(n_atoms):
            residual += construct_X(Z[k:k + 1], ds[k:k + 1])

            if sample_weights is not None:
                # with sample_weights, lhs is not toeplitz
                lhs = np.zeros((n_times_atom, n_times_atom))
            else:
                # lhs_c is only the first column of the full toeplitz matrix
                lhs_c = np.zeros(n_times_atom)

            rhs = np.zeros(n_times_atom)
            for i in range(n_trials):
                Zki[n_times_atom - 1:-(n_times_atom - 1)] = Z[k, i]
                ZZi = _embed(Zki, n_times_atom).copy()
                # n_times_atom, n_times = ZZik.shape

                if sample_weights is not None:
                    wZZi = sample_weights[i] * ZZi
                    lhs += np.dot(wZZi, ZZi.T)
                    rhs += np.dot(wZZi, residual[i])
                else:
                    lhs_c += np.dot(ZZi[0], ZZi.T)
                    rhs += np.dot(ZZi, residual[i])

            if sample_weights is None:
                # transforming the column into a squared toeplitz matrix
                lhs = linalg.toeplitz(lhs_c)

            if projection == 'primal':
                factr = solver_kwargs.get('factr', 1e11)  # default value
                d_hat = solve_unit_norm_primal(lhs, rhs, d_hat0=ds[k][::-1],
                                               factr=factr, verbose=verbose)

            elif projection == 'dual':
                factr = solver_kwargs.get('factr', 1e7)  # default value
                d_hat, lambd_hat = solve_unit_norm_dual(
                    lhs, rhs, factr=factr, lambd0=np.array([lambd0[k]]),
                    lhs_is_toeplitz=sample_weights is None)

                lambd_hats[k] = lambd_hat
            else:
                raise ValueError('Unknown projection %s.' % projection)
            ds[k] = d_hat[::-1]  # reversal for convolution
            residual -= construct_X(Z[k:k + 1], ds[k:k + 1])
    return ds, lambd_hats


def _embed(x, dim, lag=1):
    """Create an embedding of array given a resulting dimension and lag.

    Parameters
    ----------
    x : array, shape (n_times - n_times_atom + 1, )
        The array for embedding.
    dim : int
        The dimension of the stride ?

    Returns
    -------
    X.T : array, shape (n_times - 2* (n_times_atom + 1))
        The embedded array
    """
    x = x.copy()
    X = np.lib.stride_tricks.as_strided(x, (len(x) - dim * lag + lag, dim),
                                        (x.strides[0], x.strides[0] * lag))
    return X.T


def solve_unit_norm_dual(lhs, rhs, lambd0, factr=1e7, debug=False,
                         lhs_is_toeplitz=False):
    if np.all(rhs == 0):
        return np.zeros(lhs.shape[0]), 0.

    n_atoms = lambd0.shape[0]
    n_times_atom = lhs.shape[0] // n_atoms

    # precompute SVD
    # U, s, V = linalg.svd(lhs)

    if lhs_is_toeplitz:
        # first column of the toeplitz matrix lhs
        lhs_c = lhs[0, :]

        # lhs will not stay toeplitz if we add different lambd on the diagonal
        assert n_atoms == 1

        def x_star(lambd):
            lambd += 1e-14  # avoid numerical issues
            # lhs_inv = np.dot(V.T / (s + np.repeat(lambd, n_times_atom)), U.T)
            # return np.dot(lhs_inv, rhs)
            lhs_c_copy = lhs_c.copy()
            lhs_c_copy[0] += lambd
            return linalg.solve_toeplitz(lhs_c_copy, rhs)

    else:

        def x_star(lambd):
            lambd += 1e-14  # avoid numerical issues
            # lhs_inv = np.dot(V.T / (s + np.repeat(lambd, n_times_atom)), U.T)
            # return np.dot(lhs_inv, rhs)
            return linalg.solve(lhs + np.diag(np.repeat(lambd, n_times_atom)),
                                rhs)

    def dual(lambd):
        x_hats = x_star(lambd)
        norms = linalg.norm(x_hats.reshape(-1, n_times_atom), axis=1)
        return (x_hats.T.dot(lhs).dot(x_hats) - 2 * rhs.T.dot(x_hats) + np.dot(
            lambd, norms ** 2 - 1.))

    def grad_dual(lambd):
        x_hats = x_star(lambd).reshape(-1, n_times_atom)
        return linalg.norm(x_hats, axis=1) ** 2 - 1.

    def func(lambd):
        return -dual(lambd)

    def grad(lambd):
        return -grad_dual(lambd)

    bounds = [(0., None) for idx in range(0, n_atoms)]
    if debug:
        assert optimize.check_grad(func, grad, lambd0) < 1e-5
    lambd_hats, _, _ = optimize.fmin_l_bfgs_b(func, x0=lambd0, fprime=grad,
                                              bounds=bounds, factr=factr)
    x_hat = x_star(lambd_hats)
    return x_hat, lambd_hats


def solve_unit_norm_primal(lhs, rhs, d_hat0, step_size=0.1, max_iter=1000,
                           factr=1e11, verbose=0):
    """Solve d_hat in the primal using projected gradient descent.

    Stop when (f^k - f^{k+1})/max({|f^k|,|f^{k+1}|,1}) <= factr * eps
    where eps is machine precision.
    """

    def func(d_hat):
        return 0.5 * d_hat.T.dot(lhs).dot(d_hat) - rhs.T.dot(d_hat)

    def grad(d_hat):
        return np.dot(lhs, d_hat) - rhs

    def project(d_hat):
        d_hat /= max(1., linalg.norm(d_hat))
        return d_hat

    eps = np.finfo(np.float64).eps
    d_hat = d_hat0.copy()
    f_last = func(d_hat)
    for ii in range(max_iter):
        d_hat -= step_size * grad(d_hat)
        d_hat = project(d_hat)
        f = func(d_hat)
        if (f_last - f) / max([abs(f), abs(f_last), 1]) <= factr * eps:
            break
        f_last = f
    if verbose > 1:
        print('%d iterations' % (ii + 1))
    return d_hat
