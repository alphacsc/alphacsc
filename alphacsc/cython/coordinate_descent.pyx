# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
cnp.import_array()

def _locally_greedy_coordinate_selection(Zi, t_start, t_end, tol):
    """Greedily select a coordiante on the segment t_start:t_end for lil matrix

    Parameters
    ----------
    Zi : scipy.sparse.lil_matrix
        The array to select the coordinate from
    t_start, t_end: int
        The segment boundaries for the coordinate selection
    tol : float
        The minimal coordinate update selected

    """
    return _select_argmax_segment(Zi.data, Zi.rows, t_start, t_end, tol)

cdef _select_argmax_segment(object[:] Zi_data,
                            object[:] Zi_rows,
                            int t_start,
                            int t_end,
                            double tol):

    cdef int k, t, i_slice
    cdef int k0 = -1, t0 = -1
    cdef double zk, dz, adz = tol
    for k, (Zk, Zk_row) in enumerate(zip(Zi_data, Zi_rows)):
        for zk, t in zip(Zk, Zk_row):
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



def update_dz_opt(Zi, beta, dz_opt, norm_D, reg, t_start, t_end):
    """Update dz_opt for Zi encoded as lil_matrix

    Parameters
    ----------
    Zi : scipy.sparse.lil_matrix
        The array to select the coordinate from
    dz_opt, beta: ndarray, shape (n_atoms, n_times_valid)
        The auxillariay variable to update
    t_start, t_end: int
        The segment boundaries for the coordinate selection

    """
    return _update_dz_opt(Zi.data, Zi.rows, beta, dz_opt, norm_D, reg, t_start,
                          t_end)


cdef _update_dz_opt(object[:] Zi_data,
                    object[:] Zi_rows,
                    cnp.ndarray[double, ndim=2] beta,
                    cnp.ndarray[double, ndim=2] dz_opt,
                    cnp.ndarray[double, ndim=1] norm_D,
                    double reg, int t_start, int t_end):
    cdef double zk, tmp
    cdef int k, t, tk

    for k, (Zk, Zk_row) in enumerate(zip(Zi_data, Zi_rows)):
        norm_Dk = norm_D[k]
        current_t = t_start
        for tk, zk in zip(Zk_row, Zk):
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
                            cnp.ndarray[list, ndim=1] Zi_data,
                            cnp.ndarray[list, ndim=1] Zi_rows,
                            cnp.ndarray[double, ndim=1] norm_Dk):
    cdef double z
    cdef int k, t

    for k, (Zk, Zk_row) in enumerate(zip(Zi_data, Zi_rows)):
        for t, z in zip(Zk_row, Zk):
            beta[k, t] -= z * norm_Dk[k]

    return beta
