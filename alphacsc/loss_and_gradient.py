# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>

import numpy as np

from .utils.convolution import numpy_convolve_uv
from .utils.convolution import tensordot_convolve
from .utils.convolution import _choose_convolve_multi
from .utils.whitening import apply_whitening
from .utils.lil import scale_z_by_atom, safe_sum, get_z_shape, is_list_of_lil
from .utils import construct_X_multi

try:
    from .other import sdtw as _sdtw
except ImportError:
    _sdtw = None


def _assert_dtw():
    if _sdtw is None:
        raise NotImplementedError("Could not import alphacsc.other.sdtw. This "
                                  "module must be compiled with cython to "
                                  "make loss='dtw' available.")


def compute_objective(X=None, X_hat=None, z_hat=None, D=None,
                      constants=None, reg=None, loss='l2', loss_params=dict()):
    """Compute the value of the objective function

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data on which to perform CSC.
    X_hat : array, shape (n_trials, n_channels, n_times)
        The current reconstructed signal.
    z_hat : array, shape (n_atoms, n_trials, n_times_valid)
        Can also be a list of n_trials LIL-sparse matrix of shape
            (n_atoms, n_times - n_times_atom + 1)
        The current activation signals for the regularization.
    constants : dict
        Constant to accelerate the computation when updating uv.
    reg : float or array, shape (n_atoms, )
        The regularization parameters. If None, no regularization is added.
        The regularization constant
    loss : str in {'l2' | 'dtw'}
        Loss function for the data-fit
    loss_params : dict
        Parameter for the loss
    """
    if loss == 'l2':
        obj = _l2_objective(X=X, X_hat=X_hat, D=D, constants=constants)
    elif loss == 'dtw':
        _assert_dtw()
        obj = _dtw_objective(X, X_hat, loss_params=loss_params)
    elif loss == 'whitening':
        ar_model = loss_params['ar_model']

        # X is assumed to be already whitened, just select the valid part
        X = X[:, :, ar_model.ordar:-ar_model.ordar]
        X_hat = apply_whitening(ar_model, X_hat, mode='valid')
        obj = _l2_objective(X=X, X_hat=X_hat, D=D, constants=constants)
    else:
        raise NotImplementedError("loss '{}' is not implemented".format(loss))

    if reg is not None:
        if isinstance(reg, (int, float)):
            obj += reg * safe_sum(z_hat)
        else:
            obj += np.sum(reg * safe_sum(z_hat, axis=(1, 2)))

    return obj


def compute_X_and_objective_multi(X, z_hat, D_hat=None, reg=None, loss='l2',
                                  loss_params=dict(), feasible_evaluation=True,
                                  uv_constraint='joint', return_X_hat=False):
    """Compute X and return the value of the objective function

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data on which to perform CSC.
    z_hat : array, shape (n_trials, n_atoms, n_times - n_times_atom + 1)
        Can also be a list of n_trials LIL-sparse matrix of shape
            (n_atoms, n_times - n_times_atom + 1)
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
        if D_hat.ndim == 2:
            D_hat = D_hat.copy()
            # project to unit norm
            from .update_d_multi import prox_uv
            D_hat, norm = prox_uv(D_hat, uv_constraint=uv_constraint,
                                  n_channels=n_channels, return_norm=True)
        else:
            D_hat = D_hat.copy()
            # project to unit norm
            from .update_d_multi import prox_d
            D_hat, norm = prox_d(D_hat, return_norm=True)

        # update z in the opposite way
        z_hat = scale_z_by_atom(z_hat, scale=norm, copy=True)

    X_hat = construct_X_multi(z_hat, D=D_hat, n_channels=n_channels)

    cost = compute_objective(X=X, X_hat=X_hat, z_hat=z_hat, reg=reg, loss=loss,
                             loss_params=loss_params)
    if return_X_hat:
        return cost, X_hat
    return cost


def compute_gradient_norm(X, z_hat, D_hat, reg, loss='l2', loss_params=dict(),
                          rank1=False, sample_weights=None):
    if X.ndim == 2:
        X = X[:, None, :]
        D_hat = D_hat[:, None, :]

    if rank1:
        grad_d = gradient_uv(uv=D_hat, X=X, z=z_hat, constants=None,
                             loss=loss, loss_params=loss_params)
    else:
        grad_d = gradient_d(D=D_hat, X=X, z=z_hat, constants=None,
                            loss=loss, loss_params=loss_params)

    grad_norm_z = 0
    for i in range(X.shape[0]):
        grad_zi = gradient_zi(X[i], z_hat[i], D=D_hat, reg=reg,
                              loss=loss, loss_params=loss_params)
        grad_norm_z += np.dot(grad_zi.ravel(), grad_zi.ravel())

    grad_norm_d = np.dot(grad_d.ravel(), grad_d.ravel())
    grad_norm = np.sqrt(grad_norm_d) + np.sqrt(grad_norm_z)

    return grad_norm


def gradient_uv(uv, X=None, z=None, constants=None, reg=None, loss='l2',
                loss_params=dict(), return_func=False, flatten=False):
    """Compute the gradient of the reconstruction loss relative to uv.

    Parameters
    ----------
    uv : array, shape (n_atoms, n_channels + n_times_atom)
        The spatial and temporal atoms
    X : array, shape (n_trials, n_channels, n_times) or None
        The data array
    z : array, shape (n_atoms, n_trials, n_times_valid) or None
        Can also be a list of n_trials LIL-sparse matrix of shape
            (n_atoms, n_times - n_times_atom + 1)
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
    if z is not None:
        assert X is not None
        n_atoms = get_z_shape(z)[1]
        n_channels = X.shape[1]
    else:
        n_atoms = constants['ztz'].shape[0]
        n_channels = constants['n_channels']

    if is_list_of_lil(z) and loss != 'l2':
        raise NotImplementedError()

    if flatten:
        uv = uv.reshape((n_atoms, -1))

    if loss == 'l2':
        cost, grad_d = _l2_gradient_d(D=uv, X=X, z=z, constants=constants)
    elif loss == 'dtw':
        _assert_dtw()
        cost, grad_d = _dtw_gradient_d(D=uv, X=X, z=z, loss_params=loss_params)
    elif loss == 'whitening':
        cost, grad_d = _whitening_gradient_d(D=uv, X=X, z=z,
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
                cost += reg * safe_sum(z)
            else:
                cost += np.sum(reg * safe_sum(z, axis=(1, 2)))
        return cost, grad

    return grad


def gradient_zi(Xi, zi, D=None, constants=None, reg=None, loss='l2',
                loss_params=dict(), return_func=False, flatten=False):
    n_atoms = D.shape[0]
    if flatten:
        zi = zi.reshape((n_atoms, -1))

    if loss == 'l2':
        cost, grad = _l2_gradient_zi(Xi, zi, D=D, return_func=return_func)
    elif loss == 'dtw':
        _assert_dtw()
        cost, grad = _dtw_gradient_zi(Xi, zi, D=D, loss_params=loss_params)
    elif loss == 'whitening':
        cost, grad = _whitening_gradient_zi(Xi, zi, D=D,
                                            loss_params=loss_params,
                                            return_func=return_func)
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


def gradient_d(D=None, X=None, z=None, constants=None, reg=None,
               loss='l2', loss_params=dict(), return_func=False,
               flatten=False):
    """Compute the gradient of the reconstruction loss relative to d.

    Parameters
    ----------
    D : array
        The atoms. Can either be full rank with shape shape
        (n_atoms, n_channels, n_times_atom) or rank 1 with
        shape shape (n_atoms, n_channels + n_times_atom)
    X : array, shape (n_trials, n_channels, n_times) or None
        The data array
    z : array, shape (n_atoms, n_trials, n_times_valid) or None
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
        if z is None:
            n_atoms = constants['ztz'].shape[0]
            n_channels = constants['n_channels']
        else:
            n_atoms = get_z_shape(z)[1]
            n_channels = X.shape[1]
        D = D.reshape((n_atoms, n_channels, -1))

    if is_list_of_lil(z) and loss != 'l2':
        raise NotImplementedError()

    if loss == 'l2':
        cost, grad_d = _l2_gradient_d(D=D, X=X, z=z, constants=constants)
    elif loss == 'dtw':
        _assert_dtw()
        cost, grad_d = _dtw_gradient_d(D=D, X=X, z=z, loss_params=loss_params)
    elif loss == 'whitening':
        cost, grad_d = _whitening_gradient_d(D=D, X=X, z=z,
                                             loss_params=loss_params)
    else:
        raise NotImplementedError("loss {} is not implemented.".format(loss))

    if flatten:
        grad_d = grad_d.ravel()

    if return_func:
        if reg is not None:
            if isinstance(reg, float):
                cost += reg * safe_sum(z)
            else:
                cost += np.dot(reg, safe_sum(z, axis=(1, 2)))
        return cost, grad_d

    return grad_d


def _dtw_objective(X, X_hat, loss_params=dict()):
    gamma = loss_params.get('gamma')
    sakoe_chiba_band = loss_params.get('sakoe_chiba_band', -1)

    n_trials = X.shape[0]
    cost = 0
    for idx in range(n_trials):
        D_X = _sdtw.distance.SquaredEuclidean(X_hat[idx].T, X[idx].T)
        sdtw = _sdtw.SoftDTW(D_X, gamma=gamma,
                             sakoe_chiba_band=sakoe_chiba_band)
        cost += sdtw.compute()

    return cost


def _dtw_gradient(X, z, D=None, loss_params=dict()):
    gamma = loss_params.get('gamma')
    sakoe_chiba_band = loss_params.get('sakoe_chiba_band', -1)

    n_trials, n_channels, n_times = X.shape
    X_hat = construct_X_multi(z, D=D, n_channels=n_channels)
    grad = np.zeros(X_hat.shape)
    cost = 0
    for idx in range(n_trials):
        D_X = _sdtw.distance.SquaredEuclidean(X_hat[idx].T, X[idx].T)
        sdtw = _sdtw.SoftDTW(D_X, gamma=gamma,
                             sakoe_chiba_band=sakoe_chiba_band)

        cost += sdtw.compute()
        grad[idx] = D_X.jacobian_product(sdtw.grad()).T

    return cost, grad


def _dtw_gradient_d(D, X=None, z=None, loss_params={}):
    cost, grad_X_hat = _dtw_gradient(X, z, D=D, loss_params=loss_params)

    return cost, _dense_transpose_convolve_z(grad_X_hat, z)


def _dtw_gradient_zi(Xi, z_i, D=None, loss_params={}):
    n_channels = Xi.shape[0]
    cost_i, grad_Xi_hat = _dtw_gradient(Xi[None], z_i[None], D=D,
                                        loss_params=loss_params)

    return cost_i, _dense_transpose_convolve_d(
        grad_Xi_hat[0], D=D, n_channels=n_channels)


def _l2_gradient_d(D, X=None, z=None, constants=None):

    if constants:
        assert D is not None
        if D.ndim == 2:
            g = numpy_convolve_uv(constants['ztz'], D)
        else:
            g = tensordot_convolve(constants['ztz'], D)
        return None, g - constants['ztX']
    else:
        n_channels = X.shape[1]
        residual = construct_X_multi(z, D=D, n_channels=n_channels) - X
        return None, _dense_transpose_convolve_z(residual, z)


def _l2_objective(X=None, X_hat=None, D=None, constants=None):

    if constants:
        # Fast compute the l2 objective when updating uv/D
        assert D is not None, "D is needed to fast compute the objective."
        if D.ndim == 2:
            # rank 1 dictionary, use uv computation
            n_channels = constants['n_channels']
            grad_d = .5 * numpy_convolve_uv(constants['ztz'], D)
            grad_d -= constants['ztX']
            cost = (grad_d * D[:, None, n_channels:]).sum(axis=2)
            cost = np.dot(cost.ravel(), D[:, :n_channels].ravel())
        else:
            grad_d = .5 * tensordot_convolve(constants['ztz'], D)
            grad_d -= constants['ztX']
            cost = (D * grad_d).sum()

        cost += .5 * constants['XtX']
        return cost

    # else, compute the l2 norm of the residual
    assert X is not None and X_hat is not None
    residual = X - X_hat
    return 0.5 * np.dot(residual.ravel(), residual.ravel())


def _l2_gradient_zi(Xi, z_i, D=None, return_func=False):
    """

    Parameters
    ----------
    Xi : array, shape (n_channels, n_times)
        The data array for one trial
    z_i : array, shape (n_atoms, n_times_valid)
        The activations
    D : array
        The current dictionary, it can have shapes:
        - (n_atoms, n_channels + n_times_atom) for rank 1 dictionary
        - (n_atoms, n_channels, n_times_atom) for full rank dictionary
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

    Dz_i = _choose_convolve_multi(z_i, D=D, n_channels=n_channels)

    # n_channels, n_times = Dz_i.shape
    if Xi is not None:
        Dz_i -= Xi

    if return_func:
        func = 0.5 * np.dot(Dz_i.ravel(), Dz_i.ravel())
    else:
        func = None

    grad = _dense_transpose_convolve_d(Dz_i, D=D, n_channels=n_channels)

    return func, grad


def _whitening_gradient(X, X_hat, loss_params, return_func=False):
    ar_model = loss_params['ar_model']

    # Xi is assumed to be already whitened, select the valid part
    X = X[:, :, ar_model.ordar:-ar_model.ordar]

    # Construct X_hat and whitten it
    X_hat = apply_whitening(ar_model, X_hat, mode='valid')
    residual = X_hat - X

    if return_func:
        func = 0.5 * np.dot(residual.ravel(), residual.ravel())
    else:
        func = None

    hTh_res = apply_whitening(ar_model, residual, reverse_ar=True,
                              mode='full')

    return hTh_res, func


def _whitening_gradient_zi(Xi, z_i, D, loss_params, return_func=False):
    n_channels, n_times = Xi.shape

    # Construct Xi_hat and compute the gradient relatively to X_hat
    Xi_hat = construct_X_multi(z_i[None], D=D, n_channels=n_channels)
    hTh_res, func = _whitening_gradient(Xi[None], Xi_hat, loss_params,
                                        return_func=return_func)

    # Use the chain rule to compute the gradient compared to z_i
    grad = _dense_transpose_convolve_d(hTh_res[0], D=D, n_channels=n_channels)
    assert grad.shape == z_i.shape

    return func, grad


def _whitening_gradient_d(D, X, z, loss_params):
    n_channels = X.shape[1]

    # Construct Xi_hat and compute the gradient relatively to X_hat
    X_hat = construct_X_multi(z, D=D, n_channels=n_channels)
    hTh_res, func = _whitening_gradient(X, X_hat, loss_params,
                                        return_func=False)

    # Use the chain rule to compute the gradient compared to D
    grad = _dense_transpose_convolve_z(hTh_res, z)

    return None, grad


def _dense_transpose_convolve_z(residual, z):
    """Convolve residual[i] with the transpose for each atom k, and return the sum

    Parameters
    ----------
    residual : array, shape (n_trials, n_channels, n_times)
    z : array, shape (n_trials, n_atoms, n_times_valid)

    Return
    ------
    grad_D : array, shape (n_atoms, n_channels, n_times_atom)

    """
    if is_list_of_lil(z):
        raise NotImplementedError()

    return np.sum([[[np.convolve(res_ip, z_ik[::-1],
                                 mode='valid')                   # n_times_atom
                     for res_ip in res_i]                        # n_channnels
                    for z_ik in z_i]                             # n_atoms
                   for z_i, res_i in zip(z, residual)], axis=0)  # n_atoms


def _dense_transpose_convolve_d(residual_i, D=None, n_channels=None):
    """Convolve residual[i] with the transpose for each atom k

    Parameters
    ----------
    residual_i : array, shape (n_channels, n_times)
    D : array, shape (n_atoms, n_channels, n_times_atom) or
               shape (n_atoms, n_channels + n_times_atom)

    Return
    ------
    grad_zi : array, shape (n_atoms, n_times_valid)

    """

    if D.ndim == 2:
        # multiply by the spatial filter u
        uR_i = np.dot(D[:, :n_channels], residual_i)

        # Now do the dot product with the transpose of D (D.T) which is
        # the conv by the reversed filter (keeping valid mode)
        return np.array([
            np.convolve(uR_ik, v_k[::-1], 'valid')
            for (uR_ik, v_k) in zip(uR_i, D[:, n_channels:])
        ])
    else:
        return np.sum([[np.correlate(res_ip, d_kp, mode='valid')
                        for res_ip, d_kp in zip(residual_i, d_k)]
                       for d_k in D], axis=1)
