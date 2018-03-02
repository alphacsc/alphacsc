import numpy as np

from alphacsc.update_z_multi import update_z_multi


def test_gradient_correctness():

    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4
    n_times_valid = n_times - n_times_atom + 1

    reg = 0

    X = np.random.randn(n_trials, n_channels, n_times)
    u = np.random.randn(n_atoms, n_channels)
    v = np.random.randn(n_atoms, n_times_atom)
    z = np.random.randn(n_atoms, n_trials, n_times_valid)

    update_z_multi(X, u, v, reg, z0=z, n_times_atom=n_times_atom,
                   solver='l_bfgs', debug=True)
