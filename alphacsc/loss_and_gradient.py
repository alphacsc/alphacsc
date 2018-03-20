import numpy as np
from numpy import convolve


from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean


from .utils import construct_X_multi, construct_X_multi_uv
from .utils.convolution import _choose_convolve_multi_uv, numpy_convolve_uv
from .update_d_multi import prox_uv


def compute_objective(X, X_hat, Z_hat=None, reg=None, loss='l2'):
    if loss == 'l2':
        obj = _l2_objective(X, X_hat)
    elif loss == 'dtw':
        obj = _dtw_objective(X, X_hat, gamma=1)
    else:
        raise NotImplementedError("loss '{}' is not implemented".format(loss))
    if reg:
        obj += reg * Z_hat.sum()

    return obj


def compute_X_and_objective_multi(X, Z_hat, uv_hat, reg=None, loss='l2',
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
    loss : 'l2' | 'dtw'
        Loss to measure the discrepency between the signal and our estimate.
    feasible_evaluation: boolean
        If feasible_evaluation is True, it first projects on the feasible set,
        i.e. norm(uv_hat) <= 1.
    uv_constraint : str in {'joint', 'separate'}
        The kind of norm constraint on the atoms:
        If 'joint', the constraint is norm([u, v]) <= 1
        If 'separate', the constraint is norm(u) <= 1 and norm(v) <= 1
    """
    n_channels = X.shape[1]

    if feasible_evaluation:
        Z_hat = Z_hat.copy()
        uv_hat = uv_hat.copy()
        # project to unit norm
        uv_hat, norm_uv = prox_uv(uv_hat, uv_constraint=uv_constraint,
                                  n_chan=n_channels, return_norm=True)
        # update z in the opposite way
        Z_hat *= norm_uv[:, None, None]

    X_hat = construct_X_multi_uv(Z_hat, uv_hat, n_channels)

    return compute_objective(X, X_hat, Z_hat, reg, loss=loss)


def gradient_uv(uv, X=None, Z=None, constants=None, reg=None, loss='l2',
                return_func=False, flatten=False):
    """Compute the gradient of the reconstruction loss relative to uv.

    Parameters
    ----------
    uv : array, shape (n_atoms, n_channels + n_times_atom)
        The spatial and temporal atoms
    X : array, shape (n_trials, n_channels, n_times) or None
        The data array
    Z : array, shape (n_atoms, n_trials, n_times_valid) or None
        The activations
    constants : dict or None
        Constant to accelerate the computation of the gradient
    reg : float or None
        The regularization constant
    loss : str in {'l2' | 'dtw'}
    return_func : boolean
        Returns also the objective function, used to speed up LBFGS solver
    flatten : boolean
        If flatten is True, takes a flatten uv input and return the gradient
        as a flatten array.

    Returns
    -------
    (func) : float
        The objective function
    grad : array, shape (n_atoms * n_times_valid)
        The gradient
    """
    if constants:
        n_chan = constants['n_chan']
    else:
        assert X is not None
        assert Z is not None
        n_chan = X.shape[1]
    if loss == 'l2':
        cost, grad_d = _l2_gradient_d(None, X, Z, constants, uv=uv,
                                      n_chan=n_chan)
    elif loss == 'dtw':
        assert 'gamma' in constants
        cost, grad_d = _dtw_gradient_d(X, Z, uv, gamma=constants['gamma'])
    else:
        raise NotImplementedError("loss {} is not implemented.".format(loss))
    grad_u = (grad_d * uv[:, None, n_chan:]).sum(axis=2)
    grad_v = (grad_d * uv[:, :n_chan, None]).sum(axis=1)

    if return_func:
        if reg is not None:
            cost += reg * Z.sum()
        return cost, np.c_[grad_u, grad_v]

    return np.c_[grad_u, grad_v]


def gradient_zi(uv, zi, Xi, constants=None, reg=None, loss='l2',
                return_func=False, flatten=False):
    n_atoms, _ = uv.shape
    if flatten:
        zi = zi.reshape((n_atoms, -1))

    if loss == 'l2':
        cost, grad = _l2_gradient_zi(uv, zi, Xi, return_func=return_func)
    elif loss == 'dtw':
        assert 'gamma' in constants
        cost, grad = _dtw_gradient_zi(Xi, zi, uv, gamma=constants['gamma'])
    else:
        raise NotImplementedError("loss {} is not implemented.".format(loss))

    if reg is not None:
        grad += reg
        if return_func:
            cost += reg * zi.sum()

    if flatten:
        grad = grad.ravel()

    if return_func:
        return cost, grad

    return grad


def _dtw_objective(X, X_hat, gamma=1.):
    n_trials = X.shape[0]
    cost = 0
    for idx in range(n_trials):
        D = SquaredEuclidean(X_hat[idx].T, X[idx].T)
        sdtw = SoftDTW(D, gamma=gamma)
        cost += sdtw.compute()

    return cost


def _dtw_gradient(X, Z_hat, uv_hat, gamma=0.1):
    n_trials, n_channels, n_times = X.shape
    X_hat = construct_X_multi_uv(Z_hat, uv_hat, n_channels)
    grad = np.zeros(X_hat.shape)
    cost = 0
    for idx in range(n_trials):
        D = SquaredEuclidean(X_hat[idx].T, X[idx].T)
        sdtw = SoftDTW(D, gamma=gamma)

        cost += sdtw.compute()
        grad[idx] = D.jacobian_product(sdtw.grad()).T

    return cost, grad


def _dtw_gradient_d(X, Z_hat, uv_hat, gamma=0.1):
    cost, grad_X_hat = _dtw_gradient(X, Z_hat, uv_hat, gamma=gamma)

    return cost, _dense_transpose_convolve(Z_hat, grad_X_hat)


def _dtw_gradient_zi(Xi, Zi, uv, gamma=0.1):
    n_channels = Xi.shape[0]
    cost_i, grad_Xi_hat = _dtw_gradient(Xi[None], Zi[:, None], uv, gamma=gamma)

    return cost_i, _dense_transpose_convolve_uv(uv, grad_Xi_hat[0], n_channels)


def _l2_gradient_d(D, X=None, Z=None, constants=None, uv=None, n_chan=None):
    if constants:
        if D is None:
            assert uv is not None
            g = numpy_convolve_uv(constants['ZtZ'], uv)
        else:
            g = tensordot_convolve(constants['ZtZ'], D)
        return None, g - constants['ZtX']
    else:
        if D is None:
            assert uv is not None and n_chan is not None
            residual = construct_X_multi_uv(Z, uv, n_chan) - X
        else:
            assert uv is None
            residual = construct_X_multi(Z, D) - X
        return None, _dense_transpose_convolve(Z, residual)


def _l2_objective(X, X_hat):
    residual = X - X_hat
    return 0.5 * np.dot(residual.ravel(), residual.ravel())


def _l2_gradient_zi(uv, zi, Xi, return_func=False):
    """

    Parameters
    ----------
    uv : array, shape (n_atoms, n_channels + n_times_atom)
        The spatial and temporal atoms
    zi : array, shape (n_atoms, n_times_valid)
        The activations
    Xi : array, shape (n_channels, n_times)
        The data array for one trial
    return_func : boolean
        Returns also the objective function, used to speed up LBFGS solver

    Returns
    -------
    (func) : float
        The objective function l2
    grad : array, shape (n_atoms, n_times_valid)
        The gradient
    """
    n_channels, _ = Xi.shape
    n_atoms, _ = uv.shape
    # zi_reshaped = zi.reshape((n_atoms, -1))

    Dzi = _choose_convolve_multi_uv(zi, uv, n_channels)
    # n_channels, n_times = Dzi.shape
    if Xi is not None:
        Dzi -= Xi

    if return_func:
        func = 0.5 * np.dot(Dzi.ravel(), Dzi.ravel())

    grad = _dense_transpose_convolve_uv(uv, Dzi, n_channels)

    if return_func:
        return func, grad
    return None, grad


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


def _dense_transpose_convolve_uv(uv, residual, n_channels):

    # multiply by the spatial filter u
    # n_atoms, n_channels = u.shape
    # n_channels, n_times = residual.shape
    uR_i = np.dot(uv[:, :n_channels], residual)

    # Now do the dot product with the transpose of D (D.T) which is
    # the conv by the reversed filter (keeping valid mode)
    # n_atoms, n_times = uDzi.shape
    # n_atoms, n_times_atom = v.shape
    # n_atoms * n_times_valid = grad.shape
    return np.array([
        np.convolve(uR_ik, v_k[::-1], 'valid')
        for (uR_ik, v_k) in zip(uR_i, uv[:, n_channels:])
    ])
