import pytest
import numpy as np

from alphacsc.update_z_multi import update_z_multi
from alphacsc.update_z_multi import compute_DtD, _coordinate_descent_idx
from alphacsc.loss_and_gradient import compute_X_and_objective_multi
from alphacsc.utils.convolution import construct_X_multi

from alphacsc.utils.compute_constants import compute_ztz, compute_ztX


@pytest.mark.parametrize('solver', ['l-bfgs', 'ista', 'fista'])
def test_update_z_multi_decrease_cost_function(solver):
    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4
    n_times_valid = n_times - n_times_atom + 1
    reg = 0

    rng = np.random.RandomState(0)
    X = rng.randn(n_trials, n_channels, n_times)
    uv = rng.randn(n_atoms, n_channels + n_times_atom)
    z = rng.randn(n_trials, n_atoms, n_times_valid)

    loss_0 = compute_X_and_objective_multi(X=X, z_hat=z, D_hat=uv, reg=reg,
                                           feasible_evaluation=False)

    z_hat, ztz, ztX = update_z_multi(X, uv, reg, z0=z, solver=solver,
                                     return_ztz=True)

    loss_1 = compute_X_and_objective_multi(X=X, z_hat=z_hat, D_hat=uv,
                                           reg=reg, feasible_evaluation=False)
    assert loss_1 < loss_0

    assert np.allclose(ztz, compute_ztz(z_hat, n_times_atom))
    assert np.allclose(ztX, compute_ztX(z_hat, X))


@pytest.mark.parametrize('solver_z', ['l-bfgs', 'lgcd'])
def test_support_least_square(solver_z):
    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4
    n_times_valid = n_times - n_times_atom + 1
    reg = 0.1

    rng = np.random.RandomState(0)
    X = rng.randn(n_trials, n_channels, n_times)
    uv = rng.randn(n_atoms, n_channels + n_times_atom)
    z = rng.randn(n_trials, n_atoms, n_times_valid)

    # The initial loss should be high
    loss_0 = compute_X_and_objective_multi(X, z_hat=z, D_hat=uv, reg=0,
                                           feasible_evaluation=False)

    # The loss after updating z should be lower
    z_hat, _, _ = update_z_multi(X, uv, reg, z0=z, solver=solver_z)
    loss_1 = compute_X_and_objective_multi(X, z_hat=z_hat, D_hat=uv, reg=0,
                                           feasible_evaluation=False)
    assert loss_1 < loss_0

    # Here we recompute z on the support of z_hat, with reg=0
    z_hat_2, _, _ = update_z_multi(X, uv, reg=0, z0=z_hat, solver=solver_z,
                                   freeze_support=True)
    loss_2 = compute_X_and_objective_multi(X, z_hat_2, uv, 0,
                                           feasible_evaluation=False)
    assert loss_2 <= loss_1 or np.isclose(loss_1, loss_2)

    # Here we recompute z with reg=0, but with no support restriction
    z_hat_3, _, _ = update_z_multi(X, uv, reg=0, z0=z_hat_2,
                                   solver=solver_z,
                                   freeze_support=True)
    loss_3 = compute_X_and_objective_multi(X, z_hat_3, uv, 0,
                                           feasible_evaluation=False)
    assert loss_3 <= loss_2 or np.isclose(loss_3, loss_2)


def test_cd():
    n_trials, n_channels, n_times = 5, 3, 100
    n_times_atom, n_atoms = 10, 4
    n_times_valid = n_times - n_times_atom + 1
    reg = 1

    rng = np.random.RandomState(0)
    uv = rng.randn(n_atoms, n_channels + n_times_atom)
    z = abs(rng.randn(n_trials, n_atoms, n_times_valid))
    z_gen = abs(rng.randn(n_trials, n_atoms, n_times_valid))
    z[z < 1] = 0
    z_gen[z_gen < 1] = 0
    z0 = z[0]

    X = construct_X_multi(z_gen, D=uv, n_channels=n_channels)

    loss_0 = compute_X_and_objective_multi(X=X, z_hat=z_gen, D_hat=uv, reg=reg,
                                           feasible_evaluation=False)

    constants = {}
    constants['DtD'] = compute_DtD(uv, n_channels)

    # Ensure that the initialization is good, by using a nearly optimal point
    # and verifying that the cost does not goes up.
    z_hat, ztz, ztX = update_z_multi(X, D=uv, reg=reg, z0=z_gen,
                                     solver="lgcd",
                                     solver_kwargs={
                                         'max_iter': 5, 'tol': 1e-5
                                     },
                                     return_ztz=True)
    assert np.allclose(ztz, compute_ztz(z_hat, n_times_atom))
    assert np.allclose(ztX, compute_ztX(z_hat, X))

    loss_1 = compute_X_and_objective_multi(X=X, z_hat=z_hat, D_hat=uv,
                                           reg=reg,
                                           feasible_evaluation=False)
    assert loss_1 <= loss_0, "Bad initialization in greedy CD."

    z_hat, pobj, times = _coordinate_descent_idx(X[0], uv, constants, reg,
                                                 debug=True, timing=True,
                                                 z0=z0, max_iter=10000)

    try:
        assert all([p1 >= p2 for p1, p2 in zip(pobj[:-1], pobj[1:])]), "oups"
    except AssertionError:
        import matplotlib.pyplot as plt
        plt.plot(pobj)
        plt.show()
        raise


def test_n_jobs_larger_than_n_trials():
    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4

    rng = np.random.RandomState(0)
    X = rng.randn(n_trials, n_channels, n_times)
    uv = rng.randn(n_atoms, n_channels + n_times_atom)

    z_hat, ztz, ztX = update_z_multi(X, uv, 0.1, n_jobs=3)
