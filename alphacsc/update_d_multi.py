# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>
import numpy as np

from .utils.compute_constants import compute_ztz, compute_ztX


def squeeze_all_except_one(X, axis=0):
    squeeze_axis = tuple(set(range(X.ndim)) - set([axis]))
    return X.squeeze(axis=squeeze_axis)


def prox_uv(uv, uv_constraint='joint', n_channels=None, return_norm=False):
    if uv_constraint == 'joint':
        norm_uv = np.maximum(1, np.linalg.norm(uv, axis=1, keepdims=True))
        uv /= norm_uv

    elif uv_constraint == 'separate':
        assert n_channels is not None
        norm_u = np.maximum(1, np.linalg.norm(uv[:, :n_channels],
                                              axis=1, keepdims=True))
        norm_v = np.maximum(1, np.linalg.norm(uv[:, n_channels:],
                                              axis=1, keepdims=True))

        uv[:, :n_channels] /= norm_u
        uv[:, n_channels:] /= norm_v
        norm_uv = norm_u * norm_v
    else:
        raise ValueError('Unknown uv_constraint: %s.' % (uv_constraint, ))

    if return_norm:
        return uv, squeeze_all_except_one(norm_uv, axis=0)
    else:
        return uv


def prox_d(D, return_norm=False):
    norm_d = np.maximum(1, np.linalg.norm(D, axis=(1, 2), keepdims=True))
    D /= norm_d

    if return_norm:
        return D, squeeze_all_except_one(norm_d, axis=0)
    else:
        return D


def _get_d_update_constants(X, z):
    n_trials, n_atoms, n_times_valid = z.shape
    n_trials, n_channels, n_times = X.shape
    n_times_atom = n_times - n_times_valid + 1

    ztX = compute_ztX(z, X)
    ztz = compute_ztz(z, n_times_atom)

    constants = {}
    constants['ztX'] = ztX
    constants['ztz'] = ztz
    constants['n_channels'] = X.shape[1]
    constants['XtX'] = np.dot(X.ravel(), X.ravel())
    return constants
