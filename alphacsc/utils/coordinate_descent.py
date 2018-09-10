import time
import numpy as np
from scipy import sparse

from .lil import is_lil
from .. import cython_code
from ..loss_and_gradient import gradient_zi
from .convolution import _choose_convolve_multi


def _coordinate_descent_idx(Xi, D, constants, reg, z0=None, max_iter=1000,
                            tol=1e-1, strategy='greedy', n_seg='auto',
                            freeze_support=False, debug=False, timing=False,
                            name="CD", verbose=0):
    """Compute the coding signal associated to Xi with coordinate descent.

    Parameters
    ----------
    Xi : array, shape (n_channels, n_times)
        The signal to encode.
    D : array
        The atoms. Can either be full rank with shape shape
        (n_atoms, n_channels, n_times_atom) or rank 1 with
        shape shape (n_atoms, n_channels + n_times_atom)
    constants : dict
        Constants containing DtD to speedup computation
    z0 : array, shape (n_atoms, n_times_valid)
        Initial estimate of the coding signal, to warm t_start the algorithm.
    tol : float
        Tolerance for the stopping criterion of the algorithm
    max_iter : int
        Maximal number of iterations run by the algorithm
    strategy : str in {'greedy' | 'random'}
        Strategy to select the updated coordinate in the CD algorithm.
    n_seg : int or 'auto'
        Number of segments used to divide the coding signal. The updates are
        performed successively on each of these segments.
    freeze_support : boolean
        If set to True, only update the coefficient that are non-zero in z0.
    debug : boolean
        Activate extra check in the algorithm to assert that we have
        implemented the correct algorithm.
    """
    if timing:
        t_start = time.time()
    n_channels, n_times = Xi.shape
    if D.ndim == 2:
        n_atoms, n_times_atom = D.shape
        n_times_atom -= n_channels
    else:
        n_atoms, n_channels, n_times_atom = D.shape
    n_times_valid = n_times - n_times_atom + 1
    t0 = n_times_atom - 1

    if z0 is None:
        z_hat = np.zeros((n_atoms, n_times_valid))
    else:
        z_hat = z0.copy()

    if n_seg == 'auto':
        if strategy == 'greedy':
            n_seg = max(n_times_valid // (2 * n_times_atom), 1)
        elif strategy in ('random', 'cyclic'):
            n_seg = 1
            n_coordinates = n_times_valid * n_atoms

    max_iter *= n_seg
    n_times_seg = n_times_valid // n_seg + 1

    def objective(zi):
        Dzi = _choose_convolve_multi(zi, D=D, n_channels=n_channels)
        Dzi -= Xi
        func = 0.5 * np.dot(Dzi.ravel(), Dzi.ravel())
        func += reg * zi.sum()
        return func

    DtD = constants["DtD"]
    norm_Dk = np.array([DtD[k, k, t0] for k in range(n_atoms)])[:, None]

    if timing:
        times = [time.time() - t_start]
        pobj = [objective(z_hat)]
        t_start = time.time()

    beta, dz_opt = _init_beta(Xi, z_hat, D, constants, reg, norm_Dk,
                              tol, use_sparse_dz=False)

    # If we freeze the support, we put dz_opt to zero outside the support of z0
    if freeze_support:
        if is_lil(z0):
            mask = z0 != 0
            mask = mask.toarray()
            mask = ~mask
        else:
            mask = z0 == 0
        dz_opt[mask] = 0

    accumulator = n_seg
    active_segs = np.array([True] * n_seg)
    i_seg = 0
    seg_bounds = [0, n_times_seg]
    t0, k0 = -1, 0
    for ii in range(int(max_iter)):
        k0, t0, dz = _select_coordinate(strategy, dz_opt, active_segs[i_seg],
                                        n_atoms, n_times_valid, n_times_seg,
                                        seg_bounds)
        if strategy in ['random', 'cyclic']:
            # accumulate on all coordinates from the stopping criterion
            if ii % n_coordinates == 0:
                accumulator = 0
            accumulator += abs(dz)

        # Update the selected coordinate and beta, only if the update is
        # greater than the convergence tolerance.
        if abs(dz) > tol:
            # update the selected coordinate
            z_hat[k0, t0] += dz

            # update beta
            beta, dz_opt, accumulator, active_segs = _update_beta(
                beta, dz_opt, accumulator, active_segs, z_hat, DtD, norm_Dk,
                dz, k0, t0, reg, tol, seg_bounds, i_seg, n_times_atom, z0,
                freeze_support, debug)

        elif active_segs[i_seg]:
            accumulator -= 1
            active_segs[i_seg] = False

        if timing and (ii % max(100, n_seg // 100) == 0):
            times.append(time.time() - t_start)
            pobj.append(objective(z_hat))
            t_start = time.time()

        # check stopping criterion
        if strategy == 'greedy':
            if accumulator == 0:
                if verbose > 10:
                    print('[{}] {} iterations'.format(name, ii + 1))
                break
        else:
            # only check at the last coordinate
            if (ii + 1) % n_coordinates == 0 and accumulator <= tol:
                if verbose > 10:
                    print('[{}] {} iterations'.format(name, ii + 1))
                break

        # increment to next segment
        i_seg += 1
        seg_bounds[0] += n_times_seg
        seg_bounds[1] += n_times_seg

        if seg_bounds[0] >= n_times_valid:
            # reset to first segment
            i_seg = 0
            seg_bounds = [0, n_times_seg]

    else:
        if verbose > 10:
            print('[{}] did not converge'.format(name))

    if timing:
        return z_hat, pobj, times
    return z_hat


def _init_beta(Xi, z_hat, D, constants, reg, norm_Dk, tol,
               use_sparse_dz=False):
    # Init beta with -DtX
    beta = gradient_zi(Xi, z_hat, D=D, reg=None, loss='l2',
                       return_func=False, constants=constants)
    if is_lil(z_hat):
        cython_code._assert_cython()
        beta = cython_code.subtract_zhat_to_beta(beta, z_hat, norm_Dk[:, 0])
    else:
        for k, t in zip(*z_hat.nonzero()):
            beta[k, t] -= z_hat[k, t] * norm_Dk[k]  # np.sum(DtD[k, k, t0])

    if is_lil(z_hat):
        n_times_valid = beta.shape[1]
        dz_opt = np.zeros(beta.shape)
        cython_code._assert_cython()
        cython_code.update_dz_opt(z_hat, beta, dz_opt, norm_Dk[:, 0], reg,
                                  t_start=0, t_end=n_times_valid)
    else:
        dz_opt = np.maximum(-beta - reg, 0) / norm_Dk - z_hat

    if use_sparse_dz:
        dz_opt[abs(dz_opt) < tol] = 0
        dz_opt = sparse.lil_matrix(dz_opt)

    return beta, dz_opt


def _update_beta(beta, dz_opt, accumulator, active_segs, z_hat, DtD, norm_Dk,
                 dz, k0, t0, reg, tol, seg_bounds, i_seg, n_times_atom, z0,
                 freeze_support, debug):
    n_atoms, n_times_valid = beta.shape

    # define the bounds for the beta update
    t_start_up = max(0, t0 - n_times_atom + 1)
    t_end_up = min(t0 + n_times_atom, n_times_valid)

    # update beta
    beta_i0 = beta[k0, t0]
    ll = t_end_up - t_start_up
    offset = max(0, n_times_atom - t0 - 1)
    beta[:, t_start_up:t_end_up] += DtD[:, k0, offset:offset + ll] * dz
    beta[k0, t0] = beta_i0

    # update dz_opt
    if is_lil(z_hat):
        cython_code._assert_cython()
        cython_code.update_dz_opt(
            z_hat, beta, dz_opt, norm_Dk[:, 0], reg, t_start_up, t_end_up)
    else:
        tmp = np.maximum(-beta[:, t_start_up:t_end_up] - reg, 0) / norm_Dk
        dz_opt[:, t_start_up:t_end_up] = tmp - z_hat[:, t_start_up:t_end_up]
    dz_opt[k0, t0] = 0

    # reunable greedy updates in the segments immediately before or after
    # if beta was update outside the segment
    t_start_seg, t_end_seg = seg_bounds
    if t_start_up < t_start_seg and not active_segs[i_seg - 1]:
        accumulator += 1
        active_segs[i_seg - 1] = True
    if t_end_up > t_end_seg and not active_segs[i_seg + 1]:
        accumulator += 1
        active_segs[i_seg + 1] = True

    # If we freeze the support, we put dz_opt to zero outside the support of z0
    if freeze_support:
        if is_lil(z0):
            mask = z0[:, t_start_up:t_end_up] != 0
            mask = mask.toarray()
            mask = ~mask
        else:
            mask = z0[:, t_start_up:t_end_up] == 0
        dz_opt[:, t_start_up:t_end_up][mask] = 0

        if debug:
            # Check that we do not changed the support while updating beta
            nnz_z0 = list(zip(*z0[:, t_start_up:t_end_up].nonzero()))
            nnz_dz = list(zip(*dz_opt[:, t_start_up:t_end_up].nonzero()))
            assert all([nnz in nnz_z0 for nnz in nnz_dz])

    return beta, dz_opt, accumulator, active_segs


def _select_coordinate(strategy, dz_opt, active_seg, n_atoms, n_times_valid,
                       n_times_seg, seg_bounds):
    # Pick a coordinate to update
    if strategy == 'random':
        k0 = np.random.randint(n_atoms)
        t0 = np.random.randint(n_times_valid)
        dz = dz_opt[k0, t0]

    elif strategy == 'cyclic':
        t0 += 1
        if t0 >= n_times_valid:
            t0 = 0
            k0 += 1
            if k0 >= n_atoms:
                k0 = 0
        dz = dz_opt[k0, t0]

    elif strategy == 'greedy':
        # if dZs[i_seg] > tol:
        t_start_seg, t_end_seg = seg_bounds
        if active_seg:
            i0 = abs(dz_opt[:, t_start_seg:t_end_seg]).argmax()
            n_times_current = min(n_times_seg, n_times_valid - t_start_seg)
            k0, t0 = np.unravel_index(i0, (n_atoms, n_times_current))
            t0 += t_start_seg
            dz = dz_opt[k0, t0]
        else:
            k0, t0, dz = None, None, 0
    else:
        raise ValueError("'The coordinate selection method should be in "
                         "{'greedy' | 'random' | 'cyclic'}. Got '%s'."
                         % (strategy, ))
    return k0, t0, dz
