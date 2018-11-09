import numba
import numpy as np


def compute_DtD(D, n_channels=None):
    """Compute the DtD matrix
    """
    if D.ndim == 2:
        return _compute_DtD_uv(D, n_channels)

    return _compute_DtD_D(D)


@numba.jit((numba.float64[:, :], numba.int64), nopython=True, cache=True)
def _compute_DtD_uv(uv, n_channels):  # pragma: no cover
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


@numba.jit((numba.float64[:, :, :],), nopython=True, cache=True)
def _compute_DtD_D(D):  # pragma: no cover
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


@numba.jit((numba.float64[:, :, :], numba.int64), nopython=True, cache=True)
def compute_ztz(z, n_times_atom):  # pragma: no cover
    """
    ztz.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
    z.shape = n_trials, n_atoms, n_times - n_times_atom + 1)
    """
    # TODO: benchmark the cross correlate function of numpy
    n_trials, n_atoms, n_times_valid = z.shape

    ztz = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    t0 = n_times_atom - 1
    for i in range(n_trials):
        for k0 in range(n_atoms):
            for k in range(n_atoms):
                for t in range(n_times_atom):
                    if t == 0:
                        ztz[k0, k, t0] += (z[i, k0] * z[i, k]).sum()
                    else:
                        ztz[k0, k, t0 + t] += (
                            z[i, k0, :-t] * z[i, k, t:]).sum()
                        ztz[k0, k, t0 - t] += (
                            z[i, k0, t:] * z[i, k, :-t]).sum()
    return ztz


def compute_ztX(z, X):
    """
    z.shape = n_trials, n_atoms, n_times - n_times_atom + 1)
    X.shape = n_trials, n_channels, n_times
    ztX.shape = n_atoms, n_channels, n_times_atom
    """
    n_trials, n_atoms, n_times_valid = z.shape
    _, n_channels, n_times = X.shape
    n_times_atom = n_times - n_times_valid + 1

    ztX = np.zeros((n_atoms, n_channels, n_times_atom))
    for n, k, t in zip(*z.nonzero()):
        ztX[k, :, :] += z[n, k, t] * X[n, :, t:t + n_times_atom]

    return ztX
