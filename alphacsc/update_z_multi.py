# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>
import time
import warnings

import numpy as np
from scipy import optimize
from joblib import Parallel, delayed


from .utils.optim import fista
from .utils.dictionary import get_D_shape
from .loss_and_gradient import gradient_zi
from .loss_and_gradient import compute_X_and_objective_multi
from .utils.validation import check_random_state
from .utils.coordinate_descent import _coordinate_descent_idx
from .utils.compute_constants import compute_DtD, compute_ztz, compute_ztX


def update_z_multi(X, D, reg, z0=None, solver='l-bfgs', solver_kwargs=dict(),
                   freeze_support=False, positive=True,
                   return_ztz=False, timing=False, n_jobs=1,
                   random_state=None, debug=False):
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
    z0 : None | array, shape (n_trials, n_atoms, n_times_valid)
        Init for z (can be used for warm restart).
    solver : 'l-bfgs' | "lgcd"
        The solver to use.
    solver_kwargs : dict
        Parameters for the solver
    freeze_support : boolean
        If True, the support of z0 is frozen.
    positive : boolean
        If True, impose positivity constraints on z.
    return_ztz : boolean
        If True, returns the constants ztz and ztX, used to compute D-updates.
    timing : boolean
        If True, returns the cost function value at each iteration and the
        time taken by each iteration for each signal.
    n_jobs : int
        The number of parallel jobs.
    random_state : None or int or RandomState
        random_state to make randomized experiments determinist. If None, no
        random_state is given. If it is an integer, it will be used to seed a
        RandomState.
    debug : bool
        If True, check the gradients.

    Returns
    -------
    z : array, shape (n_trials, n_atoms, n_times - n_times_atom + 1)
        The true codes.
    """
    n_trials, n_channels, n_times = X.shape
    n_atoms, n_channels, n_times_atom = get_D_shape(D, n_channels)
    n_times_valid = n_times - n_times_atom + 1

    # Generate different seeds for the parallel updates
    rng = check_random_state(random_state)
    parallel_seeds = [rng.randint(2**31 - 1) for _ in range(n_trials)]

    if z0 is None:
        z0 = np.zeros((n_trials, n_atoms, n_times_valid))

    # now estimate the codes
    delayed_update_z = delayed(_update_z_multi_idx)

    results = Parallel(n_jobs=n_jobs)(
        delayed_update_z(
            X[i], D, reg, z0[i], debug, solver, solver_kwargs, freeze_support,
            positive=positive, return_ztz=return_ztz, timing=timing,
            random_state=seed,
        ) for i, seed in enumerate(parallel_seeds)
    )

    # Post process the results to get separate objects
    z_hats, pobj, times = [], [], []
    if return_ztz:
        ztz = np.zeros((n_atoms, n_atoms, 2 * n_times_atom - 1))
        ztX = np.zeros((n_atoms, n_channels, n_times_atom))
    else:
        ztz, ztX = None, None
    for z_hat, ztz_i, ztX_i, pobj_i, times_i in results:
        z_hats.append(z_hat), pobj.append(pobj_i), times.append(times_i)
        if return_ztz:
            ztz += ztz_i
            ztX += ztX_i

    # stack and reorder the columns
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


def _update_z_multi_idx(X_i, D, reg, z0_i, debug, solver='l-bfgs',
                        solver_kwargs=dict(), freeze_support=False,
                        positive=True, return_ztz=False, timing=False,
                        random_state=None):
    t_start = time.time()
    n_channels, n_times = X_i.shape
    n_atoms, n_channels, n_times_atom = get_D_shape(D, n_channels)
    n_times_valid = n_times - n_times_atom + 1

    assert not (freeze_support and z0_i is None), 'Impossible !'

    rng = check_random_state(random_state)

    constants = {}
    if solver in ["lgcd", "dicodile"]:
        constants['DtD'] = compute_DtD(D=D, n_channels=n_channels)
    init_timing = time.time() - t_start

    if z0_i is None:
        z0_i = np.zeros(n_atoms, n_times_valid)

    # Makes sure they are defined even if timing=False
    times, pobj = None, None

    if solver == 'l-bfgs':

        assert positive, "l-BFGS-B can only be used with positive=True"

        def func_and_grad(zi):
            return gradient_zi(Xi=X_i, zi=zi, D=D, constants=constants,
                               reg=reg, return_func=True, flatten=True)

        z0_i = z0_i.ravel()
        if freeze_support:
            bounds = [(0, 0) if z == 0 else (0, None) for z in z0_i]
        else:
            bounds = BoundGenerator(n_atoms * n_times_valid)
        if timing:
            times = [init_timing]
            pobj = [func_and_grad(z0_i)[0]]
            t_start = [time.time()]

            def callback(xk):
                times.append(time.time() - t_start[0])
                pobj.append(func_and_grad(xk)[0])
                # use a reference to have access inside this function
                t_start[0] = time.time()
        else:
            callback = None

        # Default values
        lbfgs_kwargs = dict(tol=1e-5, max_iter=15000, verbose=0)
        lbfgs_kwargs.update(solver_kwargs)

        # Remap parameters to l-BFGS parameters
        if 'maxiter' in solver_kwargs:
            warnings.warn(
                "maxiter for l-BFGS solver has been deprecated. "
                "Please use max_iter.", DeprecationWarning
            )
        else:
            lbfgs_kwargs['maxiter'] = lbfgs_kwargs['max_iter']
        del lbfgs_kwargs['max_iter']

        if 'factr' in solver_kwargs:
            warnings.warn(
                "factr for l-BFGS solver has been deprecated. "
                "Please use tol, which corresponds to factr * float.eps.",
                DeprecationWarning
            )
        else:
            lbfgs_kwargs['factr'] = lbfgs_kwargs['tol'] / np.finfo(float).eps
        del lbfgs_kwargs['tol']

        lbfgs_kwargs['disp'] = lbfgs_kwargs['verbose']
        del lbfgs_kwargs['verbose']

        z_hat, f, d = optimize.fmin_l_bfgs_b(
            func_and_grad, x0=z0_i, fprime=None, args=(), approx_grad=False,
            bounds=bounds, callback=callback, **lbfgs_kwargs
        )

    elif solver in ("ista", "fista"):
        # Default values
        fista_kwargs = dict(
            max_iter=15000, tol=1e-2, momentum=(solver == "fista"),
            step_size=None, adaptive_step_size=True, scipy_line_search=False,
            verbose=0,
        )
        fista_kwargs.update(solver_kwargs)

        # Remap parameters to FISTA parameters
        # XXX: rename `eps` -> `tol` in FISTA code?
        if 'eps' in solver_kwargs:
            warnings.warn(
                "eps for FISTA solver has been deprecated. "
                "Please use tol instead.", DeprecationWarning
            )
        else:
            fista_kwargs['eps'] = fista_kwargs['tol']
        del fista_kwargs['tol']

        def objective(zi):
            zi = zi.reshape(1, n_atoms, -1)
            return compute_X_and_objective_multi(
                X=X_i[None], z_hat=zi, D_hat=D, reg=reg,
                feasible_evaluation=False
            )

        def grad(zi):
            return gradient_zi(
                Xi=X_i, zi=zi, D=D, constants=constants, flatten=True
            )

        if positive:
            def prox(z_hat, step_size=0):
                return np.maximum(z_hat - step_size * reg, 0.)
        else:
            def prox(z_hat, step_size=0):
                mu = reg * step_size
                return z_hat - np.clip(z_hat, -mu, mu)

        z0_i = z0_i.ravel()
        output = fista(
            objective, grad, prox, x0=z0_i, timing=timing,
            name="Update z", **fista_kwargs
        )
        if timing:
            z_hat, pobj, times = output
            times[0] += init_timing
        else:
            z_hat, pobj = output

    elif solver in ["lgcd", "dicodile"]:

        # Default values
        lgcd_kwargs = dict(
            max_iter=1e15, tol=1e-3, n_seg='auto', strategy="greedy"
        )
        lgcd_kwargs.update(solver_kwargs)
        output = _coordinate_descent_idx(
            X_i, D, constants, reg=reg, z0=z0_i, freeze_support=freeze_support,
            positive=positive, timing=timing, random_state=rng,
            name="Update z", **lgcd_kwargs
        )

        if timing:
            z_hat, pobj, times = output
            times[0] += init_timing
        else:
            z_hat = output
    else:
        raise ValueError("Unrecognized solver %s. Must be 'ista', 'fista',"
                         " 'l-bfgs', or 'lgcd'." % solver)

    z_hat = z_hat.reshape(n_atoms, n_times_valid)

    if return_ztz:
        ztz = compute_ztz(z_hat[None], n_times_atom)
        ztX = compute_ztX(z_hat[None], X_i[None])
    else:
        ztz, ztX = None, None

    return z_hat, ztz, ztX, pobj, times
