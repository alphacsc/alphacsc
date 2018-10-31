from copy import deepcopy

import numpy as np
from scipy import sparse


def convert_to_list_of_lil(z):
    return [sparse.lil_matrix(zi) for zi in z]


def convert_from_list_of_lil(z_lil):
    return np.array([zi_lil.toarray() for zi_lil in z_lil])


def get_z_shape(z):
    if is_list_of_lil(z):
        n_trials = len(z)
        n_atoms, n_times_valid = z[0].shape
    else:
        n_trials, n_atoms, n_times_valid = z.shape
    return n_trials, n_atoms, n_times_valid


def is_list_of_lil(z):
    return isinstance(z, list) and sparse.isspmatrix_lil(z[0])


def is_lil(z):
    return sparse.isspmatrix_lil(z)


def add_one_atom_in_z(z):
    n_trials, n_atoms, n_times_valid = get_z_shape(z)

    if is_list_of_lil(z):
        def add_a_zero_line(zi_lil):
            n_atoms, n_times_valid = zi_lil.shape
            new_z = sparse.lil_matrix(np.zeros((1, n_times_valid)))
            return sparse.vstack([zi_lil, new_z])

        return [add_a_zero_line(zi_lil) for zi_lil in z]
    else:
        new_z = np.zeros((n_trials, 1, n_times_valid))
        return np.concatenate([z, new_z], axis=1)


def get_nnz_and_size(z_hat):
    if is_list_of_lil(z_hat):
        z_nnz = np.array([[len(d) for d in z.data] for z in z_hat]
                         ).sum(axis=0)
        z_size = len(z_hat) * np.prod(z_hat[0].shape)
    else:
        z_nnz = np.sum(z_hat != 0, axis=(0, 2))
        z_size = z_hat.size
    return z_nnz, z_size


def init_zeros(use_sparse_z, n_trials, n_atoms, n_times_valid):
    if use_sparse_z:
        from ..cython_code import _assert_cython
        _assert_cython()
        z_hat = [sparse.lil_matrix((n_atoms, n_times_valid))
                 for _ in range(n_trials)]
    else:
        z_hat = np.zeros((n_trials, n_atoms, n_times_valid))

    return z_hat


def scale_z_by_atom(z, scale, copy=True):
    """
    Parameters
    ----------
    z_ : array, shape (n_trials, n_atoms, n_times - n_times_atom + 1)
        Can also be a list of n_trials LIL-sparse matrix of shape
            (n_atoms, n_times - n_times_atom + 1)
        The sparse activation matrix.
    scale : array, shape = (n_atoms, )
        The scales to apply on z.
    """
    if is_list_of_lil(z):
        n_trials, n_atoms, n_times_valid = get_z_shape(z)
        assert n_atoms == len(scale)

        if copy:
            z = deepcopy(z)

        for z_i in z:
            for k in range(z_i.shape[0]):
                z_i.data[k] = [zikt * scale[k] for zikt in z_i.data[k]]

    else:
        if copy:
            z = z.copy()
        z *= scale[None, :, None]

    return z


def safe_sum(z, axis=None):
    n_trials, n_atoms, n_times_valid = get_z_shape(z)
    if is_list_of_lil(z):
        # n_trials = len(z) and (n_atoms, n_times_valid) = z[0].shape
        if axis is None:
            return sum([z_i.sum() for z_i in z])

        axis = list(axis)
        axis.sort()

        if axis == [0, 2]:
            res = np.zeros(n_atoms)
            for z_i in z:
                res += np.squeeze(np.array(z_i.sum(axis=1)))
            return res
        else:
            raise NotImplementedError()
    else:
        # (n_trials, n_atoms, n_times_valid) = z.shape
        return z.sum(axis=axis)
