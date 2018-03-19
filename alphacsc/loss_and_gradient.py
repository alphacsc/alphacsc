import numpy as np
from numba import jit
from numpy import convolve


from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean


from alphacsc.utils import construct_X_multi, construct_X_multi_uv


def gradient_uv(uv, X=None, Z=None, constants=None, loss='l2'):
    if constants:
        n_chan = constants['n_chan']
    else:
        assert X is not None
        assert Z is not None
        n_chan = X.shape[1]
    if loss == 'l2':
        grad_d = _gradient_d(None, X, Z, constants, uv=uv, n_chan=n_chan)
    else:
        assert 'gamma' in constants
        grad_d = _dtw_gradient_d(X, Z, uv, gamma=constants['gamma'])
    grad_u = (grad_d * uv[:, None, n_chan:]).sum(axis=2)
    grad_v = (grad_d * uv[:, :n_chan, None]).sum(axis=1)
    return np.c_[grad_u, grad_v]


def _dtw_objective(X, Z_hat, uv_hat, gamma=1.):
    n_trials, n_channels, n_times = X.shape
    X_hat = construct_X_multi_uv(Z_hat, uv_hat, n_channels)
    cost = 0
    for idx in range(n_trials):
        D = SquaredEuclidean(X_hat[idx].T, X[idx].T)
        sdtw = SoftDTW(D, gamma=gamma)
        # soft-DTW discrepancy, approaches DTW as gamma -> 0
        cost += sdtw.compute()

    return cost / 2.


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

    return cost / 2., grad


def _dtw_gradient_d(X, Z_hat, uv_hat, gamma=0.1):    
    cost, grad_X_hat = _dtw_gradient(X, Z_hat, uv_hat, gamma=gamma)

    return _dense_transpose_convolve(Z_hat, grad_X_hat)


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


@jit()
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
            residual = construct_X_multi_uv(Z, uv, n_chan) - X
        else:
            assert uv is None
            residual = construct_X_multi(Z, D) - X
        return _dense_transpose_convolve(Z, residual)

