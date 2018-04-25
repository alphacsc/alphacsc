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



def update_dz_opt(Zi, tmp, dz_opt, t_start, t_end):
    """Update dz_opt for Zi encoded as lil_matrix

    Parameters
    ----------
    Zi : scipy.sparse.lil_matrix
        The array to select the coordinate from
    dz_opt, tmp: ndarray, shape (n_atoms, n_times_valid)
        The auxillariay variable to update
    t_start, t_end: int
        The segment boundaries for the coordinate selection

    """
    return _update_dz_opt(Zi.data, Zi.rows, tmp, dz_opt, t_start, t_end)


cdef _update_dz_opt(object[:] Zi_data,
                    object[:] Zi_rows,
                    cnp.ndarray[double, ndim=2] tmp,
                    cnp.ndarray[double, ndim=2] dz_opt,
                    int t_start, int t_end):
    cdef double zk
    cdef int k, t, tk

    for k, (Zk, Zk_row) in enumerate(zip(Zi_data, Zi_rows)):
        current_t = t_start
        for tk, zk in zip(Zk_row, Zk):
            if tk < t_start:
                continue
            elif tk >= t_end:
                break
            for t in range(current_t, tk):
                dz_opt[k, t] = tmp[k, t - t_start]
            dz_opt[k, tk] = tmp[k, tk - t_start] - zk
            current_t = tk + 1
        for t in range(current_t, t_end):
            dz_opt[k, t] = tmp[k, t - t_start]
    return dz_opt


