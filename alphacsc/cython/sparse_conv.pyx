# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
cnp.import_array()

from alphacsc.utils import get_D


cdef _sparse_conv_d(object[:] Zi_datas,
                    object[:] Zi_rows,
                    cnp.ndarray[double, ndim=3] D,
                    cnp.ndarray[double, ndim=2] Xi):

    cdef int n_channels = D.shape[1]
    cdef int n_times_atom = D.shape[2]
    assert n_channels == Xi.shape[0]

    for dk, zk, tk in zip(D, Zi_datas, Zi_rows):
        for zkt, t in zip(zk, tk):
            Xi[:, t:t + n_times_atom] += zkt * dk

    return Xi


cdef _sparse_conv_uv(object[:] Zi_datas,
                     object[:] Zi_rows,
                     cnp.ndarray[double, ndim=2] u,
                     cnp.ndarray[double, ndim=2] v,
                     cnp.ndarray[double, ndim=2] Xi):

    cdef int n_channels = u.shape[1]
    cdef int n_times = Xi.shape[1]
    cdef int n_times_atom = v.shape[1]
    assert n_channels == Xi.shape[0]

    for uk, vk, zk, tk in zip(u, v, Zi_datas, Zi_rows):
        zik_vk = np.zeros(n_times)
        for zkt, t in zip(zk, tk):
            zik_vk[t:t + n_times_atom] += zkt * vk

        Xi += zik_vk[None, :] * uk[:, None]

    return Xi


def _fast_sparse_convolve_multi(zi_lil, ds):
    n_atoms, n_channels, n_times_atom = ds.shape
    assert zi_lil.shape[0] == n_atoms
    n_atoms, n_times_valid = zi_lil.shape
    n_times = n_times_valid + n_times_atom - 1
    Xi = np.zeros((n_channels, n_times))
    Xi = _sparse_conv_d(zi_lil.data, zi_lil.rows, ds, Xi)
    return Xi


def _fast_sparse_convolve_multi_uv(zi_lil, uv, n_channels, compute_D=True):
    n_atoms, n_channels_n_times_atom = uv.shape
    n_times_atom = n_channels_n_times_atom - n_channels
    assert zi_lil.shape[0] == n_atoms
    n_atoms, n_times_valid = zi_lil.shape
    n_times = n_times_valid + n_times_atom - 1
    Xi = np.zeros((n_channels, n_times))
    if compute_D:
        D = get_D(uv, n_channels)
        return _sparse_conv_d(zi_lil.data, zi_lil.rows, D, Xi)
    else:
        u = uv[:, :n_channels]
        v = uv[:, n_channels:]
        return _sparse_conv_uv(zi_lil.data, zi_lil.rows, u, v, Xi)
