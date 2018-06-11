import numpy as np

from .compat import numba, jit


def compute_DtD(D, n_channels=None):
    """Compute the DtD matrix
    """
    if D.ndim == 2:
        return _compute_DtD_uv(D, n_channels)

    return _compute_DtD_D(D)


@jit((numba.float64[:, :], numba.int64), nopython=True, cache=True)
def _compute_DtD_uv(uv, n_channels):
    # TODO: benchmark the cross correlate function of numpy
    n_atoms, n_times_atom = uv.shape
    n_times_atom -= n_channels

    u = uv[:, :n_channels]

    DtD = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    t0 = n_times_atom - 1
    for k0 in range(n_atoms):
        for k in range(n_atoms):
            for t in range(n_times_atom):
                if t == 0:
                    DtD[k0, k, t0] = np.dot(uv[k0, n_channels:],
                                            uv[k, n_channels:])
                else:
                    DtD[k0, k, t0 + t] = np.dot(uv[k0, n_channels:-t],
                                                uv[k, n_channels + t:])
                    DtD[k0, k, t0 - t] = np.dot(uv[k0, n_channels + t:],
                                                uv[k, n_channels:-t])
    DtD *= np.dot(u, u.T).reshape(n_atoms, n_atoms, 1)
    return DtD


@jit((numba.float64[:, :, :],), nopython=True, cache=True)
def _compute_DtD_D(D):
    # TODO: benchmark the cross correlate function of numpy
    n_atoms, n_channels, n_times_atom = D.shape

    DtD = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    t0 = n_times_atom - 1
    for k0 in range(n_atoms):
        for k in range(n_atoms):
            for t in range(n_times_atom):
                if t == 0:
                    DtD[k0, k, t0] = np.dot(D[k0].ravel(), D[k].ravel())
                else:
                    DtD[k0, k, t0 + t] = np.dot(D[k0, :, :-t].ravel(),
                                                D[k, :, t:].ravel())
                    DtD[k0, k, t0 - t] = np.dot(D[k0, :, t:].ravel(),
                                                D[k, :, :-t].ravel())
    return DtD


@jit((numba.float64[:, :, :], numba.int64), nopython=True, cache=True)
def compute_ZtZ(Z, n_times_atom):
    """
    ZtZ.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
    Z.shape = n_trials, n_atoms, n_times - n_times_atom + 1)
    """
    # TODO: benchmark the cross correlate function of numpy
    n_trials, n_atoms, n_times_valid = Z.shape

    ZtZ = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    t0 = n_times_atom - 1
    for i in range(n_trials):
        for k0 in range(n_atoms):
            for k in range(n_atoms):
                for t in range(n_times_atom):
                    if t == 0:
                        ZtZ[k0, k, t0] += (Z[i, k0] * Z[i, k]).sum()
                    else:
                        ZtZ[k0, k, t0 + t] += (
                            Z[i, k0, :-t] * Z[i, k, t:]).sum()
                        ZtZ[k0, k, t0 - t] += (
                            Z[i, k0, t:] * Z[i, k, :-t]).sum()
    return ZtZ


def compute_ZtX(Z, X):
    """
    Z.shape = n_trials, n_atoms, n_times - n_times_atom + 1)
    X.shape = n_trials, n_channels, n_times
    ZtX.shape = n_atoms, n_channels, n_times_atom
    """
    n_trials, n_atoms, n_times_valid = Z.shape
    _, n_channels, n_times = X.shape
    n_times_atom = n_times - n_times_valid + 1

    ZtX = np.zeros((n_atoms, n_channels, n_times_atom))
    for n, k, t in zip(*Z.nonzero()):
        ZtX[k, :, :] += Z[n, k, t] * X[n, :, t:t + n_times_atom]

    return ZtX
