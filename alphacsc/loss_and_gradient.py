# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>

import numpy as np

from .utils.convolution import numpy_convolve_uv
from .utils.convolution import tensordot_convolve
from .utils.convolution import _choose_convolve_multi
from .utils.convolution import construct_X_multi


def compute_objective(X=None, X_hat=None, z_hat=None, D=None,
                      constants=None, reg=None):
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
    """
    obj = _l2_objective(X=X, X_hat=X_hat, D=D, constants=constants)

    if reg is not None:
        if isinstance(reg, (int, float)):
            obj += reg * abs(z_hat).sum()
        else:
            obj += np.sum(reg * abs(z_hat).sum(axis=(1, 2)))

    return obj


def compute_X_and_objective_multi(X, z_hat, D_hat=None, reg=None,
                                  feasible_evaluation=True,
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
        z_hat = z_hat * norm[None, :, None]

    X_hat = construct_X_multi(z_hat, D=D_hat, n_channels=n_channels)

    cost = compute_objective(X=X, X_hat=X_hat, z_hat=z_hat, reg=reg)
    if return_X_hat:
        return cost, X_hat
    return cost


def compute_gradient_norm(X, z_hat, D_hat, reg,
                          rank1=False, sample_weights=None):
    if X.ndim == 2:
        X = X[:, None, :]
        D_hat = D_hat[:, None, :]

    if rank1:
        grad_d = gradient_uv(uv=D_hat, X=X, z=z_hat, constants=None)
    else:
        grad_d = gradient_d(D=D_hat, X=X, z=z_hat, constants=None)

    grad_norm_z = 0
    for i in range(X.shape[0]):
        grad_zi = gradient_zi(X[i], z_hat[i], D=D_hat, reg=reg)
        grad_norm_z += np.dot(grad_zi.ravel(), grad_zi.ravel())

    grad_norm_d = np.dot(grad_d.ravel(), grad_d.ravel())
    grad_norm = np.sqrt(grad_norm_d) + np.sqrt(grad_norm_z)

    return grad_norm


def gradient_uv(uv, X=None, z=None, constants=None, reg=None,
                return_func=False, flatten=False):
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
        n_atoms = z.shape[1]
        n_channels = X.shape[1]
    else:
        n_atoms = constants['ztz'].shape[0]
        n_channels = constants['n_channels']

    if flatten:
        uv = uv.reshape((n_atoms, -1))

    cost, grad_d = _l2_gradient_d(D=uv, X=X, z=z, constants=constants)
    grad_u = (grad_d * uv[:, None, n_channels:]).sum(axis=2)
    grad_v = (grad_d * uv[:, :n_channels, None]).sum(axis=1)
    grad = np.c_[grad_u, grad_v]

    if flatten:
        grad = grad.ravel()

    if return_func:
        if reg is not None:
            if isinstance(reg, float):
                cost += reg * abs(z).sum()
            else:
                cost += np.sum(reg * abs(z).sum(axis=(1, 2)))
        return cost, grad

    return grad


def gradient_zi(Xi, zi, D=None, constants=None, reg=None,
                return_func=False, flatten=False):
    n_atoms = D.shape[0]
    if flatten:
        zi = zi.reshape((n_atoms, -1))

    cost, grad = _l2_gradient_zi(Xi, zi, D=D, return_func=return_func)

    if reg is not None:
        grad += reg
        if return_func:
            if isinstance(reg, float):
                cost += reg * abs(zi).sum()
            else:
                cost += np.sum(reg * abs(zi).sum(axis=1))

    if flatten:
        grad = grad.ravel()

    if return_func:
        return cost, grad

    return grad


def gradient_d(D=None, X=None, z=None, constants=None, reg=None,
               return_func=False, flatten=False):
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
            n_atoms = z.shape[1]
            n_channels = X.shape[1]
        D = D.reshape((n_atoms, n_channels, -1))

    cost, grad_d = _l2_gradient_d(D=D, X=X, z=z, constants=constants)

    if flatten:
        grad_d = grad_d.ravel()

    if return_func:
        if reg is not None:
            if isinstance(reg, float):
                cost += reg * z.sum()
            else:
                cost += np.dot(reg, z.sum(axis=(1, 2)))
        return cost, grad_d

    return grad_d


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


def _dense_transpose_convolve_z(residual, z):
    """Convolve residual[i] with the transpose for each atom k, and return the
    sum for all the atoms.

    Parameters
    ----------
    residual : array, shape (n_trials, n_channels, n_times)
    z : array, shape (n_trials, n_atoms, n_times_valid)

    Return
    ------
    grad_D : array, shape (n_atoms, n_channels, n_times_atom)

    """
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
