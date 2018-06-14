# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
cnp.import_array()


cdef compute_ztX(object[:] zi_data,
                 object[:] zi_rows,
                 cnp.ndarray[double, ndim=2] Xi,
                 cnp.ndarray[double, ndim=3] ztX):

    cdef int k, t
    cdef double z
    cdef int n_times_atom = ztX.shape[2]

    for k, (zk, tk) in enumerate(zip(zi_data, zi_rows)):
        for z, t in zip(zk, tk):
            ztX[k, :, :] += z * Xi[:, t:t + n_times_atom]

    return ztX


def _fast_compute_ztX(z, X):
    """
    z.shape = n_atoms, n_trials, n_times - n_times_atom + 1)
    X.shape = n_trials, n_channels, n_times
    ztX.shape = n_atoms, n_channels, n_times_atom
    """
    n_trials, n_channels, n_times = X.shape
    assert n_trials == len(z)
    n_atoms, n_times_valid = z[0].shape
    n_times_atom = n_times - n_times_valid + 1
    ztX = np.zeros(shape=(n_atoms, n_channels, n_times_atom))
    for zi, Xi in zip(z, X):
        ztX = compute_ztX(zi.data, zi.rows, Xi, ztX)

    return ztX
