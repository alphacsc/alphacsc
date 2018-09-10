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
