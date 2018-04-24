# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
cnp.import_array()


cdef compute_ztx(object[:] Zi_data,
                 object[:] Zi_rows,
                 cnp.ndarray[double, ndim=2] Xi,
                 cnp.ndarray[double, ndim=3] ZtX):

    cdef int k, t
    cdef double z
    cdef int n_times_atom = ZtX.shape[2]

    for k, (zk, tk) in enumerate(zip(Zi_data, Zi_rows)):
        for z, t in zip(zk, tk):
            ZtX[k, :, :] += z * Xi[:, t:t + n_times_atom]

    return ZtX


def _fast_compute_ztx(Z, X):
    """
    Z.shape = n_atoms, n_trials, n_times - n_times_atom + 1)
    X.shape = n_trials, n_channels, n_times
    ZtX.shape = n_atoms, n_channels, n_times_atom
    """
    n_trials, n_channels, n_times = X.shape
    assert n_trials == len(Z)
    n_atoms, n_times_valid = Z[0].shape
    n_times_atom = n_times - n_times_valid + 1
    ZtX = np.zeros(shape=(n_atoms, n_channels, n_times_atom))
    for Zi, Xi in zip(Z, X):
        ZtX = compute_ztx(Zi.data, Zi.rows, Xi, ZtX)

    return ZtX
