# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
cnp.import_array()

def _locally_greedy_coordinate_selection(z_i, t_start, t_end, tol):
    """Greedily select a coordiante on the segment t_start:t_end for lil matrix

    Parameters
    ----------
    z_i : scipy.sparse.lil_matrix
        The array to select the coordinate from
    t_start, t_end: int
        The segment boundaries for the coordinate selection
    tol : float
        The minimal coordinate update selected

    """
    return _select_argmax_segment(z_i.data, z_i.rows, t_start, t_end, tol)

cdef _select_argmax_segment(object[:] z_i_data,
                            object[:] z_i_rows,
                            int t_start,
                            int t_end,
                            double tol):

    cdef int k, t, i_slice
    cdef int k0 = -1, t0 = -1
    cdef double zk, dz, adz = tol
    for k, (z_k, z_k_row) in enumerate(zip(z_i_data, z_i_rows)):
        for zk, t in zip(z_k, z_k_row):
            if t < t_start:
                continue
            if t >= t_end:
                break
            if abs(zk) > adz:
                k0 = k
                t0 = t
                adz = abs(zk)
                dz = zk

    return k0, t0, dz



def update_dz_opt(z_i, beta, dz_opt, norm_D, reg, t_start, t_end):
    """Update dz_opt for z_i encoded as lil_matrix

    Parameters
    ----------
    z_i : scipy.sparse.lil_matrix
        The array to select the coordinate from
    dz_opt, beta: ndarray, shape (n_atoms, n_times_valid)
        The auxillariay variable to update
    t_start, t_end: int
        The segment boundaries for the coordinate selection

    """
    return _update_dz_opt(z_i.data, z_i.rows, beta, dz_opt, norm_D, reg, t_start,
                          t_end)


cdef _update_dz_opt(object[:] z_i_data,
                    object[:] z_i_rows,
                    cnp.ndarray[double, ndim=2] beta,
                    cnp.ndarray[double, ndim=2] dz_opt,
                    cnp.ndarray[double, ndim=1] norm_D,
                    double reg, int t_start, int t_end):
    cdef double zk, tmp
    cdef int k, t, tk

    for k, (z_k, z_k_row) in enumerate(zip(z_i_data, z_i_rows)):
        norm_Dk = norm_D[k]
        current_t = t_start
        for tk, zk in zip(z_k_row, z_k):
            if tk < t_start:
                continue
            elif tk >= t_end:
                break
            for t in range(current_t, tk):
                tmp = max(-beta[k, t] - reg, 0) / norm_Dk
                dz_opt[k, t] = tmp
            tmp = max(-beta[k, tk] - reg, 0) / norm_Dk
            dz_opt[k, tk] = tmp - zk
            current_t = tk + 1
        for t in range(current_t, t_end):
            tmp = max(-beta[k, t] - reg, 0) / norm_Dk
            dz_opt[k, t] = tmp
    return dz_opt


def subtract_zhat_to_beta(beta, z_hat, norm_Dk):
    """Equivalent to:

    for k, t in zip(*z_hat.nonzero()):
        beta[k, t] -= z_hat[k, t] * norm_Dk[k]
    """
    return _subtract_zhat_to_beta(beta, z_hat.data, z_hat.rows, norm_Dk)


cdef _subtract_zhat_to_beta(cnp.ndarray[double, ndim=2] beta,
                            cnp.ndarray[list, ndim=1] z_i_data,
                            cnp.ndarray[list, ndim=1] z_i_rows,
                            cnp.ndarray[double, ndim=1] norm_Dk):
    cdef double z
    cdef int k, t

    for k, (z_k, z_k_row) in enumerate(zip(z_i_data, z_i_rows)):
        for t, z in zip(z_k_row, z_k):
            beta[k, t] -= z * norm_Dk[k]

    return beta
