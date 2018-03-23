import numpy as np

from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean


from .utils import construct_X_multi, construct_X_multi_uv
from .utils.convolution import _choose_convolve_multi_uv, numpy_convolve_uv
from .utils.convolution import tensordot_convolve


def compute_objective(X=None, X_hat=None, Z_hat=None, uv=None, constants=None,
                      reg=None, loss='l2', loss_params=dict()):
    """Compute the value of the objective function

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data on which to perform CSC.
    X_hat : array, shape (n_trials, n_channels, n_times)
        The current reconstructed signal.
    Z_hat : array, shape (n_atoms, n_trial, n_times_valid)
        The current activation signals for the regularization.
    constants : dict
        Constant to accelerate the computation when updating uv.
    reg : float
        The regularization parameters. If None, no regularization is added.
        The regularization constant
    loss : str in {'l2' | 'dtw'}
        Loss function for the data-fit
    loss_params : dict
        Parameter for the loss
    """
    if loss == 'l2':
        obj = _l2_objective(X=X, X_hat=X_hat, uv=uv, constants=constants)
    elif loss == 'dtw':
        obj = _dtw_objective(X, X_hat, loss_params=loss_params)
    else:
        raise NotImplementedError("loss '{}' is not implemented".format(loss))

    if reg is not None:
        if isinstance(reg, float):
            obj += reg * Z_hat.sum()
        else:
            obj += np.sum(reg * Z_hat.sum(axis=(1, 2)))

    return obj


def compute_X_and_objective_multi(X, Z_hat, uv_hat, reg=None, loss='l2',
                                  loss_params=dict(), feasible_evaluation=True,
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
    loss_params : dict
        Parameters of the loss
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
        from .update_d_multi import prox_uv
        uv_hat, norm_uv = prox_uv(uv_hat, uv_constraint=uv_constraint,
                                  n_chan=n_channels, return_norm=True)
        # update z in the opposite way
        Z_hat *= norm_uv[:, None, None]

    X_hat = construct_X_multi_uv(Z_hat, uv_hat, n_channels)

    return compute_objective(X=X, X_hat=X_hat, Z_hat=Z_hat, reg=reg, loss=loss,
                             loss_params=loss_params)


def gradient_uv(uv, X=None, Z=None, constants=None, reg=None, loss='l2',
                loss_params=dict(), return_func=False, flatten=False):
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
        Loss function for the data-fit
    loss_params : dict
        Parameter for the loss
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
    if Z is not None:
        assert X is not None
        n_atoms = Z.shape[0]
        n_channels = X.shape[1]
    else:
        n_atoms = constants['ZtZ'].shape[0]
        n_channels = constants['n_chan']

    if flatten:
        uv = uv.reshape((n_atoms, -1))

    if loss == 'l2':
        cost, grad_d = _l2_gradient_d(uv=uv, X=X, Z=Z, constants=constants)
    elif loss == 'dtw':
        cost, grad_d = _dtw_gradient_d(X=X, Z=Z, uv=uv,
                                       loss_params=loss_params)
    else:
        raise NotImplementedError("loss {} is not implemented.".format(loss))
    grad_u = (grad_d * uv[:, None, n_channels:]).sum(axis=2)
    grad_v = (grad_d * uv[:, :n_channels, None]).sum(axis=1)
    grad = np.c_[grad_u, grad_v]

    if flatten:
        grad = grad.ravel()

    if return_func:
        if reg is not None:
            if isinstance(reg, float):
                cost += reg * Z.sum()
            else:
                cost += np.sum(reg * Z.sum(axis=(1, 2)))
        return cost, grad

    return grad


def gradient_zi(uv, zi, Xi, constants=None, reg=None, loss='l2',
                loss_params=dict(), return_func=False, flatten=False):
    n_atoms, _ = uv.shape
    if flatten:
        zi = zi.reshape((n_atoms, -1))

    if loss == 'l2':
        cost, grad = _l2_gradient_zi(uv, zi, Xi, return_func=return_func)
    elif loss == 'dtw':
        cost, grad = _dtw_gradient_zi(Xi, zi, uv, loss_params=loss_params)
    else:
        raise NotImplementedError("loss {} is not implemented.".format(loss))

    if reg is not None:
        grad += reg
        if return_func:
            if isinstance(reg, float):
                cost += reg * zi.sum()
            else:
                cost += np.sum(reg * zi.sum(axis=1))

    if flatten:
        grad = grad.ravel()

    if return_func:
        return cost, grad

    return grad


def gradient_d(D=None, uv=None, X=None, Z=None, constants=None, reg=None,
               loss='l2', loss_params=dict(), return_func=False,
               flatten=False):
    """Compute the gradient of the reconstruction loss relative to d.

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
        Loss function for the data-fit
    loss_params : dict
        Parameter for the loss
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
    if flatten:
        n_atoms = Z.shape[0]
        n_channels = X.shape[1]
        D = D.reshape((n_atoms, n_channels, -1))
    if loss == 'l2':
        cost, grad_d = _l2_gradient_d(D=D, uv=uv, X=X, Z=Z,
                                      constants=constants)
    elif loss == 'dtw':
        cost, grad_d = _dtw_gradient_d(D=D, X=X, Z=Z, uv=uv,
                                       loss_params=loss_params)
    else:
        raise NotImplementedError("loss {} is not implemented.".format(loss))

    if flatten:
        grad_d = grad_d.ravel()

    if return_func:
        if reg is not None:
            if isinstance(reg, float):
                cost += reg * Z.sum()
            else:
                cost += np.dot(reg, Z.sum(axis=(1, 2)))
        return cost, grad_d

    return grad_d


def _dtw_objective(X, X_hat, loss_params=dict()):
    gamma = loss_params.get('gamma')
    sakoe_chiba_band = loss_params.get('sakoe_chiba_band', -1)

    n_trials = X.shape[0]
    cost = 0
    for idx in range(n_trials):
        D = SquaredEuclidean(X_hat[idx].T, X[idx].T)
        sdtw = SoftDTW(D, gamma=gamma, sakoe_chiba_band=sakoe_chiba_band)
        cost += sdtw.compute()

    return cost


def _dtw_gradient(X, Z, uv=None, D=None, loss_params=dict()):
    gamma = loss_params.get('gamma')
    sakoe_chiba_band = loss_params.get('sakoe_chiba_band', -1)

    n_trials, n_channels, n_times = X.shape
    if uv is not None:
        X_hat = construct_X_multi_uv(Z, uv, n_channels)
    else:
        X_hat = construct_X_multi(Z, D)
    grad = np.zeros(X_hat.shape)
    cost = 0
    for idx in range(n_trials):
        D = SquaredEuclidean(X_hat[idx].T, X[idx].T)
        sdtw = SoftDTW(D, gamma=gamma, sakoe_chiba_band=sakoe_chiba_band)

        cost += sdtw.compute()
        grad[idx] = D.jacobian_product(sdtw.grad()).T

    return cost, grad


def _dtw_gradient_d(X=None, Z=None, uv=None, D=None, loss_params=None):
    cost, grad_X_hat = _dtw_gradient(X, Z, uv=uv, D=D, loss_params=loss_params)

    return cost, _dense_transpose_convolve(Z, grad_X_hat)


def _dtw_gradient_zi(Xi, Zi, uv, loss_params):
    n_channels = Xi.shape[0]
    cost_i, grad_Xi_hat = _dtw_gradient(Xi[None], Zi[:, None], uv,
                                        loss_params=loss_params)

    return cost_i, _dense_transpose_convolve_uv(uv, grad_Xi_hat[0], n_channels)


def _l2_gradient_d(D=None, uv=None, X=None, Z=None, constants=None):

    if constants:
        if D is None:
            assert uv is not None
            g = numpy_convolve_uv(constants['ZtZ'], uv)
        else:
            g = tensordot_convolve(constants['ZtZ'], D)
        return None, g - constants['ZtX']
    else:
        if D is None:
            n_channels = X.shape[1]
            assert uv is not None
            residual = construct_X_multi_uv(Z, uv, n_channels) - X
        else:
            assert uv is None
            residual = construct_X_multi(Z, D) - X
        return None, _dense_transpose_convolve(Z, residual)


def _l2_objective(X=None, X_hat=None, uv=None, constants=None):

    if constants:
        # Fast compute the l2 objective when updating uv
        assert uv is not None
        n_chan = constants['n_chan']
        grad_d = .5 * numpy_convolve_uv(constants['ZtZ'], uv)
        grad_d -= constants['ZtX']
        cost = (grad_d * uv[:, None, n_chan:]).sum(axis=2)
        cost = np.dot(cost.ravel(), uv[:, :n_chan].ravel())
        cost += .5 * constants['XtX']
        return cost

    # else, compute the l2 norm of the residual
    assert X is not None and X_hat is not None
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
    return np.sum([[[np.convolve(res_ip, zik[::-1],
                                 mode='valid')  # n_times_atom
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
