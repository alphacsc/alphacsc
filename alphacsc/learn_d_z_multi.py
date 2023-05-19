# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>

from __future__ import print_function
import time
import sys

import numpy as np

from .utils.validation import check_dimension
from .utils.validation import check_random_state
from .utils.convolution import sort_atoms_by_explained_variances
from ._z_encoder import get_z_encoder_for
from ._d_solver import get_solver_d


def learn_d_z_multi(X, n_atoms, n_times_atom, n_iter=60, n_jobs=1,
                    lmbd_max='fixed', reg=0.1,
                    rank1=True, uv_constraint='auto', eps=1e-10,
                    algorithm='batch', algorithm_params=dict(),
                    solver_z='l-bfgs', solver_z_kwargs=dict(),
                    solver_d='auto', solver_d_kwargs=dict(),
                    D_init=None,
                    unbiased_z_hat=False, stopping_pobj=None,
                    raise_on_increase=True, verbose=10, callback=None,
                    random_state=None, name="DL", window=False,
                    sort_atoms=False):
    """Multivariate Convolutional Sparse Coding with optional rank-1 constraint

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data on which to perform CSC.
    n_atoms : int
        The number of atoms to learn.
    n_times_atom : int
        The support of the atom.
    reg : float
        The regularization parameter
    lmbd_max : 'fixed' | 'scaled' | 'per_atom' | 'shared'
        If not fixed, adapt the regularization rate as a ratio of lambda_max:
          - 'scaled': the regularization parameter is fixed as a ratio of its
            maximal value at init i.e. reg_used = reg * lmbd_max(uv_init)
          - 'shared': the regularization parameter is set at each iteration as
            a ratio of its maximal value for the current dictionary estimate
            i.e. reg_used = reg * lmbd_max(uv_hat)
          - 'per_atom': the regularization parameter is set per atom and at
            each iteration as a ratio of its maximal value for this atom i.e.
            reg_used[k] = reg * lmbd_max(uv_hat[k])
    n_iter : int
        The number of coordinate-descent iterations.
    n_jobs : int
        The number of parallel jobs.

    rank1 : boolean
        If set to True, learn rank 1 dictionary atoms.
        If solver_z is 'dicodile', then rank1 must be False.
    uv_constraint : str in {'joint' | 'separate' | 'auto'}
        The kind of norm constraint on the atoms if using rank1=True.
        If 'joint', the constraint is norm_2([u, v]) <= 1
        If 'separate', the constraint is norm_2(u) <= 1 and norm_2(v) <= 1
        If rank1 is False, then uv_constraint must be 'auto'.
    eps : float
        Stopping criterion. If the cost descent after a uv and a z update is
        smaller than eps, return.
    algorithm : 'batch' | 'greedy' | 'online' | 'stochastic'
        Dictionary learning algorithm.
    algorithm_params : dict
        Parameters for the global algorithm used to learn the dictionary:
          alpha : float
            Forgetting factor for online learning. If set to 0, the learning is
            stochastic and each D-step is independent from the previous steps.
            When set to 1, each the previous values z_hat - computed with
            different dictionary - have the same weight as the current one.
            This factor should be large enough to ensure convergence but to
            large factor can lead to sub-optimal minima.
          batch_selection : 'random' | 'cyclic'
            The batch selection strategy for online learning. The batch are
            either selected randomly among all samples (without replacement) or
            in a cyclic way.
          batch_size : int in [1, n_trials]
            Size of the batch used in online learning. Increasing it
            regularizes the dictionary learning as there is less variance for
            the successive estimates. But it also increases the computational
            cost as more coding signals z_hat must be estimate at each
            iteration.
    solver_z : str
        The solver to use for the z update. Options are
        'l-bfgs' (default) | 'lgcd' |
        'dicodile' (distributed LGCD, experimental)
    solver_z_kwargs : dict
        Additional keyword arguments to pass to update_z_multi
    solver_d : str
        The solver to use for the d update. If rank1 is False, only option is
        'fista'. Else, options are 'alternate', 'alternate_adaptive' (default)
        or 'joint'.
    solver_d_kwargs : dict
        Additional keyword arguments to provide to update_d
    D_init : str or array, shape (n_atoms, n_channels + n_times_atoms) or \
                           shape (n_atoms, n_channels, n_times_atom)
        The initial atoms or an initialization scheme in {'chunk' | 'random' |
        'greedy'}.
    unbiased_z_hat : boolean
        If set to True, the value of the non-zero coefficients in the returned
        z_hat are recomputed with reg=0 on the frozen support.
    verbose : int
        The verbosity level.
    callback : func
        A callback function called at the end of each loop of the
        coordinate descent, with z_encoder and pobj as its arguments.
    random_state : int | None
        The random state.
    raise_on_increase : boolean
        Raise an error if the objective function increase
    window : boolean
        If True, re-parametrizes the atoms with a temporal Tukey window
    sort_atoms : boolean
        If True, the atoms are sorted by explained variances.

    Returns
    -------
    pobj : list
        The objective function value at each step of the coordinate descent.
    times : list
        The cumulative time for each iteration of the coordinate descent.
    uv_hat : array, shape (n_atoms, n_channels + n_times_atom)
        The atoms to learn from the data.
    z_hat : array, shape (n_trials, n_atoms, n_times_valid)
        The sparse activation matrix.
    reg : float
        Regularization parameter used.
    """

    assert lmbd_max in ['fixed', 'scaled', 'per_atom', 'shared'], (
        "lmbd_max should be in {'fixed', 'scaled', 'per_atom', 'shared'}, "
        f"got '{lmbd_max}'"
    )

    _, n_channels, _ = check_dimension(X)

    # Rescale the problem to avoid underflow issues
    std_X = X.std()
    X = X / std_X

    if algorithm == "stochastic":
        # The typical stochastic algorithm samples one signal, compute the
        # associated value z and then perform one step of gradient descent
        # for D.
        assert (
            'max_iter' not in solver_d_kwargs or solver_d_kwargs['max_iter'] != 1  # noqa
        ), (
            "with algorithm='stochastic', max_iter is forced to 1."
        )
        solver_d_kwargs["max_iter"] = 1

    elif algorithm == 'greedy':
        # Initialize D with no atoms as they will be added sequentially.
        D_init = 'greedy'

    # initialization
    start = time.time()

    d_solver = get_solver_d(
        n_channels, n_atoms, n_times_atom, solver_d=solver_d, rank1=rank1,
        uv_constraint=uv_constraint, D_init=D_init, window=window,
        random_state=random_state, **solver_d_kwargs
    )

    D_hat = d_solver.init_dictionary(X)

    init_duration = time.time() - start

    z_kwargs = dict(verbose=verbose, **solver_z_kwargs)

    with get_z_encoder_for(
            X, d_solver.D_hat, n_atoms, n_times_atom, n_jobs,
            solver_z, z_kwargs, reg
    ) as z_encoder:

        if callable(callback):
            callback(z_encoder, [])

        end_iter_func = get_iteration_func(
            eps, stopping_pobj, callback, lmbd_max,
            name, verbose, raise_on_increase
        )

        # common parameters
        kwargs = dict(
            z_encoder=z_encoder, d_solver=d_solver, n_iter=n_iter,
            end_iter_func=end_iter_func, lmbd_max=lmbd_max,
            verbose=verbose, random_state=random_state, name=name
        )
        kwargs.update(algorithm_params)

        if algorithm == 'batch':
            pobj, times = _batch_learn(greedy=False, **kwargs)
        elif algorithm == "greedy":
            pobj, times = _batch_learn(greedy=True, **kwargs)
        elif algorithm == "online":
            pobj, times = _online_learn(**kwargs)
        elif algorithm == "stochastic":
            # For stochastic learning, set forgetting factor alpha of the
            # online algorithm to 0, making each step independent of previous
            # steps and set D-update max_iter to a low value (typically 1).
            kwargs['alpha'] = 0
            pobj, times = _online_learn(**kwargs)
        else:
            raise NotImplementedError(
                "Algorithm '{}' is not implemented to learn dictionary atoms."
                .format(algorithm))

        D_hat = d_solver.D_hat
        z_hat = z_encoder.get_z_hat()

        if sort_atoms:
            D_hat, z_hat = sort_atoms_by_explained_variances(
                D_hat, z_hat, n_channels=n_channels)

        # recompute z_hat with no regularization and keeping the support fixed
        if unbiased_z_hat:
            start_unbiased_z_hat = time.time()
            z_encoder.compute_z(unbiased_z_hat=True)
            z_hat = z_encoder.get_z_hat()
            if verbose > 1:
                print(
                    "[{}] Compute the final z_hat with support freeze in "
                    "{:.2f}s".format(name, time.time() - start_unbiased_z_hat))

        times[0] += init_duration

        if verbose > 0:
            print("[%s] Fit in %.1fs" % (name, time.time() - start))

        # Rescale the solution to match the given scale of the problem
        z_hat *= std_X
        reg = z_encoder.reg * std_X

    return pobj, times, D_hat, z_hat, reg


def _batch_learn(z_encoder, d_solver, end_iter_func, n_iter=100,
                 lmbd_max='fixed', reg=None, verbose=0, greedy=False,
                 random_state=None, name="batch"):

    _, n_atoms, _ = z_hat_shape = z_encoder.get_z_hat_shape()

    if greedy:
        n_iter_by_atom = 1

        if n_iter < n_atoms * n_iter_by_atom:
            raise ValueError('The greedy method needs at least %d iterations '
                             'to learn %d atoms. Got only n_iter=%d. Please '
                             'increase n_iter.' % (
                                 n_iter_by_atom * n_atoms, n_atoms, n_iter))

    # monitor cost function
    times = [0]
    pobj = [z_encoder.get_cost()]

    for ii in range(n_iter):  # outer loop of coordinate descent
        if verbose == 1:
            msg = '.' if ((ii + 1) % 50 != 0) else '+\n'
            print(msg, end='')
            sys.stdout.flush()
        if verbose > 1:
            print('[{}] CD iterations {} / {}'.format(name, ii, n_iter))

        if greedy and ii % n_iter_by_atom == 0 and \
                d_solver.D_hat.shape[0] < n_atoms:
            # add a new atom every n_iter_by_atom iterations
            d_solver.add_one_atom(z_encoder)

        if lmbd_max in ['per_atom', 'shared'] or (
                lmbd_max == 'scaled' and ii == 0
        ):
            z_encoder.update_reg(lmbd_max == 'per_atom')

            if verbose > 5:
                print('[{}] lambda = {:.3e}'.format(name,
                                                    np.mean(z_encoder.reg)))

        # Compute z update
        start = time.time()
        z_encoder.compute_z()

        # monitor cost function
        times.append(time.time() - start)
        pobj.append(z_encoder.get_cost())

        z_nnz = z_encoder.get_z_nnz()
        if verbose > 5:
            print(
                f"[{name}] Objective (z) : {pobj[-1]:.3e} "
                f"(sparsity: {z_nnz.sum() / np.prod(z_hat_shape):.3e})"
            )

        if np.all(z_nnz == 0):
            import warnings
            warnings.warn("Regularization parameter `reg` is too large "
                          "and all the activations are zero. No atom has"
                          " been learned.", UserWarning)
            break

        # Compute D update
        start = time.time()
        d_solver.update_D(z_encoder)

        # monitor cost function
        times.append(time.time() - start)
        pobj.append(z_encoder.get_cost())

        null_atom_indices = np.where(z_nnz < 2)[0]
        if len(null_atom_indices) > 0:
            k0 = null_atom_indices[0]
            d_solver.resample_atom(k0, z_encoder)

            if verbose > 5:
                print('[{}] Resampled atom {}'.format(name, k0))

        if verbose > 5:
            print('[{}] Objective (d) : {:.3e}'.format(name, pobj[-1]))

        if ((not greedy or d_solver.D_hat.shape[0] == n_atoms)
                and end_iter_func(z_encoder, pobj, ii)):
            break

    return pobj, times


def _online_learn(z_encoder, d_solver, end_iter_func, n_iter=100,
                  verbose=0, random_state=None, lmbd_max='fixed', reg=None,
                  alpha=.8, batch_selection='random', batch_size=1,
                  name="online"):

    n_trials, *_ = z_hat_shape = z_encoder.get_z_hat_shape()

    # monitor cost function
    times = [0]
    pobj = [z_encoder.get_cost()]

    rng = check_random_state(random_state)
    for ii in range(n_iter):  # outer loop of coordinate descent
        if verbose == 1:
            msg = '.' if (ii % 50 != 0) else '+\n'
            print(msg, end='')
            sys.stdout.flush()
        if verbose > 1:
            print('[{}] CD iterations {} / {}'.format(name, ii, n_iter))

        if lmbd_max in ['per_atom', 'shared'] or (
                lmbd_max == 'scaled' and ii == 0
        ):
            z_encoder.update_reg(lmbd_max == 'per_atom')

            if verbose > 5:
                print('[{}] lambda = {:.3e}'.format(name,
                                                    np.mean(z_encoder.reg)))

        # Compute z update
        start = time.time()
        if batch_selection == 'random':
            i0 = rng.choice(n_trials, batch_size, replace=False)
        elif batch_selection == 'cyclic':
            i_slice = (ii * batch_size) % n_trials
            i0 = slice(i_slice, i_slice + batch_size)
        else:
            raise NotImplementedError(
                "the '{}' batch_selection strategy for the online learning is "
                "not implemented.".format(batch_selection))
        z_encoder.compute_z_partial(i0, alpha)

        # monitor cost function
        times.append(time.time() - start)
        pobj.append(z_encoder.get_cost())

        z_nnz = z_encoder.get_z_nnz()
        if verbose > 5:
            print(
                f"[{name}] Objective (z) : {pobj[-1]:.3e} "
                f"(sparsity: {z_nnz.sum() / np.prod(z_hat_shape):.3e})"
            )

        if np.all(z_nnz == 0):
            import warnings
            warnings.warn("Regularization parameter `reg` is too large "
                          "and all the activations are zero. No atoms has"
                          " been learned.", UserWarning)
            break

        # Compute D update
        start = time.time()
        d_solver.update_D(z_encoder)

        # monitor cost function
        times.append(time.time() - start)
        pobj.append(z_encoder.get_cost())

        null_atom_indices = np.where(z_nnz < 2)[0]
        if len(null_atom_indices) > 0:
            k0 = null_atom_indices[0]
            d_solver.resample_atom(k0, z_encoder)
            if verbose > 5:
                print('[{}] Resampled atom {}'.format(name, k0))

        if verbose > 5:
            print('[{}] Objective (d) : {:.3e}'.format(name, pobj[-1]))

        if end_iter_func(z_encoder, pobj, ii):
            break

    return pobj, times


def get_iteration_func(eps, stopping_pobj, callback, lmbd_max, name, verbose,
                       raise_on_increase):
    def end_iteration(z_encoder, pobj, iteration):
        if callable(callback):
            callback(z_encoder, pobj)

        # Only check that the cost is always going down when the regularization
        # parameter is fixed.
        dz = (pobj[-3] - pobj[-2]) / min(pobj[-3], pobj[-2])
        du = (pobj[-2] - pobj[-1]) / min(pobj[-2], pobj[-1])
        if ((dz < eps or du < eps) and lmbd_max in ['fixed', 'scaled']):
            if dz < 0 and raise_on_increase:
                raise RuntimeError(
                    "The z update have increased the objective value by {}."
                    .format(dz))
            if du < -1e-10 and dz > 1e-12 and raise_on_increase:
                raise RuntimeError(
                    "The d update have increased the objective value by {}."
                    "(dz={})".format(du, dz))
            if dz < eps and du < eps:
                if verbose == 1:
                    print("")
                print("[{}] Converged after {} iteration, (dz, du) "
                      "= {:.3e}, {:.3e}".format(name, iteration + 1, dz, du))
                return True

        if stopping_pobj is not None and pobj[-1] < stopping_pobj:
            return True
        return False

    return end_iteration
