"""Convolutional dictionary learning"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>
import time

import numpy as np
from scipy import optimize, sparse
from joblib import Parallel, delayed


from .loss_and_gradient import gradient_zi
from .utils.optim import fista
from .utils.lil import is_list_of_lil, is_lil
from .utils.convolution import _choose_convolve_multi
from .cython import _fast_compute_ztz, _fast_compute_ztX
from .utils.compute_constants import compute_DtD, compute_ztz, compute_ztX
from .cython.coordinate_descent import update_dz_opt, subtract_zhat_to_beta


def update_z_multi(X, D, reg, z0=None, debug=False, parallel=None,
                   solver='l_bfgs', solver_kwargs=dict(), loss='l2',
                   loss_params=dict(), freeze_support=False, timing=False):
    """Update z using L-BFGS with positivity constraints

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data array
    D : array, shape (n_atoms, n_channels + n_times_atom)
        The dictionary used to encode the signal X. Can be either in the form
        f a full rank dictionary D (n_atoms, n_channels, n_times_atom) or with
        the spatial and temporal atoms uv (n_atoms, n_channels + n_times_atom).
    reg : float
        The regularization constant
    z0 : None | array, shape (n_trials, n_atoms, n_times_valid) |
         list of sparse lil_matrices, shape (n_atoms, n_times_valid)
        Init for z (can be used for warm restart).
    debug : bool
        If True, check the grad.
    parallel : instance of Parallel
        Context manager for running joblibs in a loop.
    solver : 'l_bfgs' | 'gcd'
        The solver to use.
    solver_kwargs : dict
        Parameters for the solver
    loss : 'l2' | 'dtw' | 'whitening'
        The data fit loss, either classical l2 norm or the soft-DTW loss.
    loss_params : dict
        Parameters of the loss
    freeze_support : boolean
        If True, the support of z0 is frozen.
    timing : boolean
        If True, returns the cost function value at each iteration and the
        time taken by each iteration for each signal.

    Returns
    -------
    z : array, shape (n_trials, n_atoms, n_times - n_times_atom + 1)
        The true codes.
    """
    n_trials, n_channels, n_times = X.shape
    if D.ndim == 2:
        n_atoms, n_channels_n_times_atom = D.shape
        n_times_atom = n_channels_n_times_atom - n_channels
    else:
        n_atoms, n_channels, n_times_atom = D.shape
    n_times_valid = n_times - n_times_atom + 1

    # now estimate the codes
    delayed_update_z = delayed(_update_z_multi_idx)
    if parallel is None:
        parallel = Parallel(n_jobs=1)

    results = parallel(
        delayed_update_z(X[i], D, reg, z0[i], debug, solver, solver_kwargs,
                         freeze_support, loss, loss_params=loss_params,
                         timing=timing)
        for i in np.arange(n_trials))

    # Initialize return args
    z_hats, pobj, times = [], [], []
    ztz = np.zeros((n_atoms, n_atoms, 2 * n_times_atom - 1))
    ztX = np.zeros((n_atoms, n_channels, n_times_atom))
    for z_hat, ztz_i, ztX_i, pobj_i, times_i in results:
        z_hats.append(z_hat), pobj.append(pobj_i), times.append(times_i)
        ztz += ztz_i
        ztX += ztX_i

    # If z_hat is a ndarray, stack and reorder the columns
    if not is_list_of_lil(z0):
        z_hats = np.array(z_hats).reshape(n_trials, n_atoms, n_times_valid)

    return z_hats, ztz, ztX


class BoundGenerator(object):
    def __init__(self, length):
        self.length = length
        self.current_index = 0

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index == self.length:
            raise StopIteration()
        self.current_index += 1
        return (0, np.inf)


def _update_z_multi_idx(X_i, D, reg, z0_i, debug, solver="l_bfgs",
                        solver_kwargs=dict(), freeze_support=False, loss='l2',
                        loss_params=dict(), timing=False):
    t_start = time.time()
    n_channels, n_times = X_i.shape
    if D.ndim == 2:
        n_atoms, n_channels_n_times_atom = D.shape
        n_times_atom = n_channels_n_times_atom - n_channels
    else:
        n_atoms, n_channels, n_times_atom = D.shape
    n_times_valid = n_times - n_times_atom + 1

    assert not (freeze_support and z0_i is None), 'Impossible !'

    if is_lil(z0_i) and solver != "gcd":
        raise NotImplementedError()

    constants = {}
    if solver == "gcd":
        constants['DtD'] = compute_DtD(D=D, n_channels=n_channels)
    init_timing = time.time() - t_start

    def func_and_grad(zi):
        return gradient_zi(Xi=X_i, zi=zi, D=D, constants=constants,
                           reg=reg, return_func=True, flatten=True,
                           loss=loss, loss_params=loss_params)

    if z0_i is None:
        f0 = np.zeros(n_atoms * n_times_valid)
    elif is_lil(z0_i):
        f0 = z0_i
    else:
        f0 = z0_i.reshape(n_atoms * n_times_valid)

    times, pobj = None, None
    if timing:
        times = [init_timing]
        pobj = [func_and_grad(f0)[0]]
        t_start = [time.time()]

    if solver == "l_bfgs":
        if freeze_support:
            bounds = [(0, 0) if z == 0 else (0, None) for z in f0]
        else:
            bounds = BoundGenerator(n_atoms * n_times_valid)
        if timing:
            def callback(xk):
                times.append(time.time() - t_start[0])
                pobj.append(func_and_grad(xk)[0])
                # use a reference to have access inside this function
                t_start[0] = time.time()
        else:
            callback = None
        factr = solver_kwargs.get('factr', 1e15)  # default value
        maxiter = solver_kwargs.get('maxiter', 15000)  # default value
        z_hat, f, d = optimize.fmin_l_bfgs_b(
            func_and_grad, f0, fprime=None, args=(), approx_grad=False,
            bounds=bounds, factr=factr, maxiter=maxiter, callback=callback)

    elif solver in ("ista", "fista"):
        # Default args
        fista_kwargs = dict(
            max_iter=100, eps=None, verbose=0, restart=None,
            scipy_line_search=False,
            momentum=(solver == "fista")
        )
        fista_kwargs.update(solver_kwargs)

        def objective(z_hat):
            return func_and_grad(z_hat)[0]

        def grad(z_hat):
            return func_and_grad(z_hat)[1]

        def prox(z_hat,):
            return np.maximum(z_hat, 0.)

        output = fista(objective, grad, prox, None, f0,
                       adaptive_step_size=True, timing=timing,
                       name="Update Z", **fista_kwargs)
        if timing:
            z_hat, pobj, times = output
            times[0] += init_timing
        else:
            z_hat, pobj = output

    elif solver == "gcd":
        if not sparse.isspmatrix_lil(f0):
            f0 = f0.reshape(n_atoms, n_times_valid)

        # Default values
        tol = solver_kwargs.get('tol', 1e-1)
        n_seg = solver_kwargs.get('n_seg', 'auto')
        max_iter = solver_kwargs.get('max_iter', 1e15)
        strategy = solver_kwargs.get('strategy', 'greedy')
        output = _coordinate_descent_idx(
            X_i, D, constants, reg=reg, z0=f0,
            freeze_support=freeze_support, tol=tol, max_iter=max_iter,
            n_seg=n_seg, strategy=strategy, timing=timing)
        if timing:
            z_hat, pobj, times = output
            times[0] += init_timing
        else:
            z_hat = output
    else:
        raise ValueError("Unrecognized solver %s. Must be 'ista', 'fista',"
                         " or 'l_bfgs'." % solver)

    if not is_lil(z_hat):

        z_hat = z_hat.reshape(n_atoms, n_times_valid)
        ztz = compute_ztz(z_hat[None], n_times_atom)
        ztX = compute_ztX(z_hat[None], X_i[None])
    else:
        ztz = _fast_compute_ztz([z_hat], n_times_atom)
        ztX = _fast_compute_ztX([z_hat], X_i[None])

    return z_hat, ztz, ztX, pobj, times


def _coordinate_descent_idx(Xi, D, constants, reg, z0=None, max_iter=1000,
                            tol=1e-1, strategy='greedy', n_seg='auto',
                            freeze_support=False, debug=False, timing=False,
                            verbose=0):
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
                break
        else:
            # only check at the last coordinate
            if (ii + 1) % n_coordinates == 0 and accumulator <= tol:
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
            print('[CD] update z did not converge')
    if verbose > 10:
        print('[CD] update z computed %d iterations' % (ii + 1))

    if timing:
        return z_hat, pobj, times
    return z_hat


def _init_beta(Xi, z_hat, D, constants, reg, norm_Dk, tol,
               use_sparse_dz=False):
    # Init beta with -DtX
    beta = gradient_zi(Xi, z_hat, D=D, reg=None, loss='l2',
                       return_func=False, constants=constants)
    if is_lil(z_hat):
        beta = subtract_zhat_to_beta(beta, z_hat, norm_Dk[:, 0])
    else:
        for k, t in zip(*z_hat.nonzero()):
            beta[k, t] -= z_hat[k, t] * norm_Dk[k]  # np.sum(DtD[k, k, t0])

    if is_lil(z_hat):
        n_times_valid = beta.shape[1]
        dz_opt = np.zeros(beta.shape)
        update_dz_opt(z_hat, beta, dz_opt, norm_Dk[:, 0], reg, t_start=0,
                      t_end=n_times_valid)
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
        update_dz_opt(z_hat, beta, dz_opt, norm_Dk[:, 0], reg, t_start_up,
                      t_end_up)
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
