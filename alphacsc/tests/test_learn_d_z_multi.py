import numpy as np

from alphacsc.learn_d_z_multi import learn_d_z_multi


def test_learn_d_z_multi():
    # smoke test for learn_d_z_multi
    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4

    X = np.random.randn(n_trials, n_channels, n_times)
    for uv_constraint in ['joint', 'separate']:
        pobj, times, uv_hat, Z_hat = learn_d_z_multi(
            X, n_atoms, n_times_atom, uv_constraint=uv_constraint,
            random_state=0, n_iter=2)
