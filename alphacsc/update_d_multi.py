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


def check_solver_and_constraints(rank1, solver_d, uv_constraint):
    """Checks if solver_d and uv_constraint are compatible depending on
    rank1 value.

    - If rank1 is False, solver_d should be 'fista' and uv_constraint should be
    'auto'.
    - If rank1 is True;
       - If solver_d is either 'alternate' or 'alternate_adaptive',
         uv_constraint should be 'separate'.
       - If solver_d is either 'joint' or 'fista', uv_constraint should be
         'joint'.

    Parameters
    ----------
    rank1: boolean
        If set to True, learn rank 1 dictionary atoms.
    solver_d : str in {'alternate' | 'alternate_adaptive' | 'fista' | 'joint' |
    'auto'}
        The solver to use for the d update.
        - If rank1 is False, only option is 'fista'
        - If rank1 is True, options are 'alternate', 'alternate_adaptive'
          (default) or 'joint'
    uv_constraint : str in {'joint' | 'separate' | 'auto'}
        The kind of norm constraint on the atoms if using rank1=True.
        If 'joint', the constraint is norm_2([u, v]) <= 1
        If 'separate', the constraint is norm_2(u) <= 1 and norm_2(v) <= 1
        If rank1 is False, then uv_constraint must be 'auto'.
    """

    if rank1:
        if solver_d == 'auto':
            solver_d = 'alternate_adaptive'
        if 'alternate' in solver_d:
            if uv_constraint == 'auto':
                uv_constraint = 'separate'
            else:
                assert uv_constraint == 'separate', (
                    "solver_d='alternate*' should be used with "
                    f"uv_constraint='separate'. Got '{uv_constraint}'."
                )
        elif uv_constraint == 'auto' and solver_d in ['joint', 'fista']:
            uv_constraint = 'joint'
    else:
        assert solver_d in ['auto', 'fista'], (
            f"solver_d should be auto or fista. Got solver_d='{solver_d}'."
        )
        assert solver_d in ['auto', 'fista'] and uv_constraint == 'auto', (
            "If rank1 is False, uv_constraint should be 'auto' "
            f"and solver_d should be auto or fista. Got solver_d='{solver_d}' "
            f"and uv_constraint='{uv_constraint}'."
        )
        solver_d = 'fista'
    return solver_d, uv_constraint


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
