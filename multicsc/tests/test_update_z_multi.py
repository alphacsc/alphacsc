import pytest
import numpy as np
from scipy import sparse

from multicsc.update_z_multi import update_z_multi
from multicsc.update_z_multi import compute_DtD, _coordinate_descent_idx
from multicsc.loss_and_gradient import compute_X_and_objective_multi
from multicsc.utils.whitening import whitening
from multicsc.utils import construct_X_multi


@pytest.mark.parametrize('loss', ['l2', 'dtw', 'whitening'])
def test_gradient_correctness(loss):

    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4
    n_times_valid = n_times - n_times_atom + 1
    reg = 0
    loss_params = dict(gamma=1, sakoe_chiba_band=n_times_atom // 2)

    X = np.random.randn(n_trials, n_channels, n_times)
    uv = np.random.randn(n_atoms, n_channels + n_times_atom)
    z = np.random.randn(n_atoms, n_trials, n_times_valid)

    if loss == 'whitening':
        loss_params['ar_model'], X = whitening(X, ordar=10)

    update_z_multi(X, uv, reg, z0=z, solver='l_bfgs', debug=True,
                   loss=loss, loss_params=loss_params)


@pytest.mark.parametrize('loss', ['l2', 'dtw', 'whitening'])
@pytest.mark.parametrize('solver', ['l_bfgs', 'ista', 'fista'])
def test_update_z_multi_decrease_cost_function(loss, solver):
    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4
    n_times_valid = n_times - n_times_atom + 1
    reg = 0
    loss_params = dict(gamma=1, sakoe_chiba_band=n_times_atom // 2)

    X = np.random.randn(n_trials, n_channels, n_times)
    uv = np.random.randn(n_atoms, n_channels + n_times_atom)
    z = np.random.randn(n_atoms, n_trials, n_times_valid)

    if loss == 'whitening':
        loss_params['ar_model'], X = whitening(X, ordar=10)

    loss_0 = compute_X_and_objective_multi(X=X, Z_hat=z, D_hat=uv, reg=reg,
                                           feasible_evaluation=False,
                                           loss=loss, loss_params=loss_params)

    z_hat = update_z_multi(X, uv, reg, z0=z, solver=solver, loss=loss,
                           loss_params=loss_params)

    loss_1 = compute_X_and_objective_multi(X=X, Z_hat=z_hat, D_hat=uv,
                                           reg=reg, feasible_evaluation=False,
                                           loss=loss, loss_params=loss_params)
    assert loss_1 < loss_0


def test_support_least_square():
    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4
    n_times_valid = n_times - n_times_atom + 1
    reg = 0

    X = np.random.randn(n_trials, n_channels, n_times)
    uv = np.random.randn(n_atoms, n_channels + n_times_atom)
    z = np.random.randn(n_atoms, n_trials, n_times_valid)

    # The initial loss should be high
    loss_0 = compute_X_and_objective_multi(X, Z_hat=z, D_hat=uv, reg=reg,
                                           feasible_evaluation=False)

    # The loss after updating z should be lower
    z_hat = update_z_multi(X, uv, reg, z0=z, solver='l_bfgs',
                           solver_kwargs={'factr': 1e7})
    loss_1 = compute_X_and_objective_multi(X, Z_hat=z_hat, D_hat=uv, reg=reg,
                                           feasible_evaluation=False)
    assert loss_1 < loss_0

    # Here we recompute z on the support of z_hat, with reg=0
    z_hat_2 = update_z_multi(X, uv, reg=0, z0=z_hat, solver='l_bfgs',
                             solver_kwargs={'factr': 1e7}, freeze_support=True)
    loss_2 = compute_X_and_objective_multi(X, z_hat_2, uv, reg,
                                           feasible_evaluation=False)
    assert loss_2 <= loss_1 or np.isclose(loss_1, loss_2)

    # Here we recompute z with reg=0, but with no support restriction
    z_hat_3 = update_z_multi(X, uv, reg=0, z0=np.ones(z.shape),
                             solver='l_bfgs', solver_kwargs={'factr': 1e7},
                             freeze_support=True)
    loss_3 = compute_X_and_objective_multi(X, z_hat_3, uv, reg,
                                           feasible_evaluation=False)
    assert loss_3 <= loss_2 or np.isclose(loss_3, loss_2)


@pytest.mark.parametrize('use_sparse_lil', [True, False])
def test_cd(use_sparse_lil):
    n_trials, n_channels, n_times = 5, 3, 100
    n_times_atom, n_atoms = 10, 4
    n_times_valid = n_times - n_times_atom + 1
    reg = 1

    uv = np.random.randn(n_atoms, n_channels + n_times_atom)
    if use_sparse_lil:
        density = .1
        Z = [sparse.random(n_atoms, n_times_valid, format='lil',
                           density=density) for _ in range(n_trials)]
        Z_gen = [sparse.random(n_atoms, n_times_valid, format='lil',
                               density=density) for _ in range(n_trials)]
        Z0 = Z[0]
    else:
        Z = abs(np.random.randn(n_atoms, n_trials, n_times_valid))
        Z_gen = abs(np.random.randn(n_atoms, n_trials, n_times_valid))
        Z[Z < 1] = 0
        Z_gen[Z_gen < 1] = 0
        Z0 = Z[:, 0]

    X = construct_X_multi(Z_gen, D=uv, n_channels=n_channels)

    loss_0 = compute_X_and_objective_multi(X=X, Z_hat=Z_gen, D_hat=uv, reg=reg,
                                           loss='l2',
                                           feasible_evaluation=False)

    constants = {}
    constants['DtD'] = compute_DtD(uv, n_channels)

    # Ensure that the initialization is good, by using a nearly optimal point
    # and verifying that the cost does not goes up.
    z_hat = update_z_multi(X, D=uv, reg=reg, z0=Z_gen,
                           solver='gcd',
                           solver_kwargs={
                               'max_iter': 5, 'tol': 1e-5}
                           )

    loss_1 = compute_X_and_objective_multi(X=X, Z_hat=z_hat, D_hat=uv,
                                           reg=reg, loss='l2',
                                           feasible_evaluation=False)
    assert loss_1 <= loss_0, "Bad initialization in greedy CD."

    z_hat, pobj = _coordinate_descent_idx(X[0], uv, constants, reg, debug=True,
                                          z0=Z0, max_iter=10000)

    try:
        assert all([p1 >= p2 for p1, p2 in zip(pobj[:-1], pobj[1:])]), "oups"
    except AssertionError:
        import matplotlib.pyplot as plt
        plt.plot(pobj)
        plt.show()
        raise
