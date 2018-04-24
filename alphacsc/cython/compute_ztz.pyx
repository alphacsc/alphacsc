# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
cnp.import_array()


cdef compute_ztz(object[:] Zi_data,
                 object[:] Zi_rows,
                 int nnz,
                 int n_times_valid,
                 cnp.ndarray[double, ndim=3] ZtZ):

    cdef int k, k0, ctk, ltk, t, t0
    cdef double z0, z
    cdef int n_atoms = len(Zi_data)
    cdef int n_times_atom = (ZtZ.shape[2] + 1) // 2

    lower_tk = np.zeros(n_atoms, dtype=int)
    current_tk = np.zeros(n_atoms, dtype=int)
    for _ in range(nnz):
        # Find the argmin of the curent tk
        k0 = 0
        t0 = n_times_valid
        for k, ctk in enumerate(current_tk):
            if ctk >= len(Zi_rows[k]):
                continue
            t = Zi_rows[k][ctk]
            if t < t0:
                k0 = k
                t0 = t
                z0 = Zi_data[k][ctk]

        # Update the pointers on lower valid indices
        for k in range(n_atoms):
            tk = Zi_rows[k]
            while (lower_tk[k] < len(tk) and
                    tk[lower_tk[k]] <= t0 - n_times_atom):
                lower_tk[k] += 1

        # compute the correlation from Z_k[t] with Z
        for k, (zk, tk, ltk) in enumerate(zip(Zi_data, Zi_rows, lower_tk)):
            for z, t in zip(zk[ltk:], tk[ltk:]):
                if t - t0 > n_times_atom - 1:
                    break
                ZtZ[k, k0, n_times_atom - 1 + t0 - t] += z0 * z

        current_tk[k0] += 1

    return ZtZ


def _fast_compute_ztz(Z, n_times_atom):
    """
    ZtZ.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
    Z = list of LIL matrices of shape = (n_atoms, n_times - n_times_atom + 1)
    """
    n_trials = len(Z)
    n_atoms, n_times_valid = Z[0].shape
    ZtZ = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    for Zi in Z:
        ZtZ = compute_ztz(Zi.data, Zi.rows, Zi.nnz, n_times_valid, ZtZ)

    return ZtZ
