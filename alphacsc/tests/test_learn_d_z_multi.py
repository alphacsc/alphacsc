import pytest
import numpy as np

from alphacsc.learn_d_z_multi import learn_d_z_multi


@pytest.mark.parametrize('solver_d, uv_constraint', [
    ('joint', 'joint'), ('alternate', 'separate'), ('lbfgs', 'box')
])
def test_learn_d_z_multi(solver_d, uv_constraint):
    # smoke test for learn_d_z_multi
    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4

    X = np.random.randn(n_trials, n_channels, n_times)
    pobj, times, uv_hat, Z_hat = learn_d_z_multi(
        X, n_atoms, n_times_atom, uv_constraint=uv_constraint,
        solver_d=solver_d, random_state=0, n_iter=10)

    msg = "Cost function does not go down for uv_constraint {}".format(
        uv_constraint)
    try:
        assert all([p1 >= p2 for p1, p2 in zip(pobj[:-1], pobj[1:])]), msg
    finally:
        import matplotlib.pyplot as plt
        plt.semilogy(pobj - np.min(pobj) + 1e-6)
        plt.show()
