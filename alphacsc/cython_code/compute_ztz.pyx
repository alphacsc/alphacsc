# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
cnp.import_array()


cdef compute_ztz_lil(object[:] z_i_data,
                     object[:] z_i_rows,
                     int nnz,
                     int n_times_valid,
                     cnp.ndarray[double, ndim=3] ztz):

    cdef int k, k0, ctk, ltk, t, t0
    cdef double z0, z
    cdef int n_atoms = len(z_i_data)
    cdef int n_times_atom = (ztz.shape[2] + 1) // 2

    lower_tk = np.zeros(n_atoms, dtype=int)
    current_tk = np.zeros(n_atoms, dtype=int)
    for _ in range(nnz):
        # Find the argmin of the curent tk
        k0 = 0
        t0 = n_times_valid
        for k, ctk in enumerate(current_tk):
            if ctk >= len(z_i_rows[k]):
                continue
            t = z_i_rows[k][ctk]
            if t < t0:
                k0 = k
                t0 = t
                z0 = z_i_data[k][ctk]

        # Update the pointers on lower valid indices
        for k in range(n_atoms):
            tk = z_i_rows[k]
            while (lower_tk[k] < len(tk) and
                    tk[lower_tk[k]] <= t0 - n_times_atom):
                lower_tk[k] += 1

        # compute the correlation from z_k[t] with z
        for k, (zk, tk, ltk) in enumerate(zip(z_i_data, z_i_rows, lower_tk)):
            for z, t in zip(zk[ltk:], tk[ltk:]):
                if t - t0 > n_times_atom - 1:
                    break
                ztz[k, k0, n_times_atom - 1 + t0 - t] += z0 * z

        current_tk[k0] += 1

    return ztz



cdef compute_ztz_csr(cnp.ndarray[double, ndim=1] z_i_data,
                     cnp.ndarray[int, ndim=1] z_i_indices,
                     cnp.ndarray[int, ndim=1] z_i_indptr,
                     cnp.ndarray[double, ndim=3] ztz):

    cdef int k, k0, i, i0, t, t0, start_k0, start_k, end_k0, end_k, idx
    cdef int n_atoms = ztz.shape[0]
    cdef int n_times_atom = (ztz.shape[2] + 1) // 2

    for k0 in range(n_atoms):
        start_k0 = z_i_indptr[k0]
        end_k0 = z_i_indptr[k0 + 1]
        for k in range(n_atoms):
            start_k = z_i_indptr[k]
            end_k = z_i_indptr[k + 1]
            for i0 in range(start_k0, end_k0):
                t0 = z_i_indices[i0]
                for i in range(start_k, end_k):
                    t = z_i_indices[i]
                    idx = n_times_atom - 1 + t0 - t
                    if 0 <= idx < 2 * n_times_atom - 1:
                        ztz[k, k0, idx] += z_i_data[i] * z_i_data[i0]
    return ztz


def _fast_compute_ztz_lil(z, n_times_atom):
    """
    ztz.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
    z = list of LIL matrices of shape = (n_atoms, n_times - n_times_atom + 1)
    """
    n_trials = len(z)
    n_atoms, n_times_valid = z[0].shape
    ztz = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    for z_i in z:
        ztz = compute_ztz_lil(z_i.data, z_i.rows, z_i.nnz, n_times_valid, ztz)

    return ztz


def _fast_compute_ztz_csr(z, n_times_atom):
    """
    ztz.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
    z = list of LIL matrices of shape = (n_atoms, n_times - n_times_atom + 1)
    """
    n_trials = len(z)
    n_atoms, n_times_valid = z[0].shape
    ztz = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    for z_i in z:
        z_i_csr = z_i.tocsr()
        ztz = compute_ztz_csr(z_i_csr.data, z_i_csr.indices, z_i_csr.indptr, ztz)

    return ztz


_fast_compute_ztz = _fast_compute_ztz_csr
