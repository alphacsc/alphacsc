import numpy as np

from alphacsc.update_z_multi import update_z_multi, _support_least_square
from alphacsc.learn_d_z_multi import compute_X_and_objective_multi


def test_gradient_correctness():

    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4
    n_times_valid = n_times - n_times_atom + 1
    reg = 0

    X = np.random.randn(n_trials, n_channels, n_times)
    uv = np.random.randn(n_atoms, n_channels + n_times_atom)
    z = np.random.randn(n_atoms, n_trials, n_times_valid)

    update_z_multi(X, uv, reg, z0=z, solver='l_bfgs', debug=True)


def test_update_z_multi_decrease_cost_function():
    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4
    n_times_valid = n_times - n_times_atom + 1
    reg = 0

    X = np.random.randn(n_trials, n_channels, n_times)
    uv = np.random.randn(n_atoms, n_channels + n_times_atom)
    z = np.random.randn(n_atoms, n_trials, n_times_valid)

    loss_0 = compute_X_and_objective_multi(X, z, uv, reg,
                                           feasible_evaluation=False)

    z_hat = update_z_multi(X, uv, reg, z0=z, solver='l_bfgs')

    loss_1 = compute_X_and_objective_multi(X, z_hat, uv, reg,
                                           feasible_evaluation=False)
    assert loss_1 < loss_0


def test_support_least_square():
    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4
    n_times_valid = n_times - n_times_atom + 1
    reg = 0

    X = np.random.randn(n_trials, n_channels, n_times)
    uv = np.random.randn(n_atoms, n_channels + n_times_atom)
    z = np.random.randn(n_atoms, n_trials, n_times_valid)

    loss_0 = compute_X_and_objective_multi(X, z, uv, reg,
                                           feasible_evaluation=False)

    z_hat = update_z_multi(X, uv, reg, z0=z, solver='l_bfgs',
                           solver_kwargs={'factr': 1})

    loss_1 = compute_X_and_objective_multi(X, z_hat, uv, reg,
                                           feasible_evaluation=False)
    assert loss_1 < loss_0

    z_hat_2 = _support_least_square(X, uv, z_hat)
    loss_2 = compute_X_and_objective_multi(X, z_hat_2, uv, reg,
                                           feasible_evaluation=False)
    assert np.isclose(loss_2, loss_1)
    assert np.allclose(z_hat, z_hat_2)

    z_hat_3 = _support_least_square(X, uv, np.ones(z.shape))
    loss_3 = compute_X_and_objective_multi(X, z_hat_3, uv, reg,
                                           feasible_evaluation=False)
    import matplotlib.pyplot as plt
    plt.plot(z_hat_2.flatten())
    plt.plot(z_hat_3.flatten())
    plt.show()
    assert np.isclose(loss_3, loss_2)
