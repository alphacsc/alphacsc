from copy import deepcopy

import numpy as np
from scipy import sparse


def convert_to_list_of_lil(Z):
    return [sparse.lil_matrix(zi) for zi in Z]


def convert_from_list_of_lil(Z_lil):
    return np.array([zi_lil.toarray() for zi_lil in Z_lil])


def get_Z_shape(Z):
    if is_list_of_lil(Z):
        n_trials = len(Z)
        n_atoms, n_times_valid = Z[0].shape
    else:
        n_trials, n_atoms, n_times_valid = Z.shape
    return n_trials, n_atoms, n_times_valid


def is_list_of_lil(Z):
    return isinstance(Z, list) and sparse.isspmatrix_lil(Z[0])


def is_lil(Z):
    return sparse.isspmatrix_lil(Z)


def scale_Z_by_atom(Z, scale, copy=True):
    """
    Parameters
    ----------
    Z_ : array, shape (n_trials, n_atoms, n_times - n_times_atom + 1)
        Can also be a list of n_trials LIL-sparse matrix of shape
            (n_atoms, n_times - n_times_atom + 1)
        The sparse activation matrix.
    scale : array, shape = (n_atoms, )
        The scales to apply on Z.
    """
    if is_list_of_lil(Z):
        n_trials, n_atoms, n_times_valid = get_Z_shape(Z)
        assert n_atoms == len(scale)

        if copy:
            Z = deepcopy(Z)

        for Zi in Z:
            for k in range(Zi.shape[0]):
                Zi.data[k] = [zikt * scale[k] for zikt in Zi.data[k]]

    else:
        if copy:
            Z = Z.copy()
        Z *= scale[None, :, None]

    return Z


def safe_sum(Z, axis=None):
    n_trials, n_atoms, n_times_valid = get_Z_shape(Z)
    if is_list_of_lil(Z):
        # n_trials = len(Z) and (n_atoms, n_times_valid) = Z[0].shape
        if axis is None:
            return sum([Zi.sum() for Zi in Z])

        axis = list(axis)
        axis.sort()

        if axis == [0, 2]:
            res = np.zeros(n_atoms)
            for Zi in Z:
                res += np.squeeze(np.array(Zi.sum(axis=1)))
            return res
        else:
            raise NotImplementedError()
    else:
        # (n_trials, n_atoms, n_times_valid) = Z.shape
        return Z.sum(axis=axis)
