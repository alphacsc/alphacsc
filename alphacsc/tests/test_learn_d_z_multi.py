import pytest
import numpy as np

from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.utils import check_random_state


@pytest.mark.parametrize('solver_d, uv_constraint', [
    ('joint', 'joint'), ('joint', 'separate'),
    ('alternate', 'separate'), ('lbfgs', 'box')
])
def test_learn_d_z_multi(solver_d, uv_constraint):
    # smoke test for learn_d_z_multi
    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4

    rng = check_random_state(42)
    X = rng.randn(n_trials, n_channels, n_times)
    pobj, times, uv_hat, Z_hat = learn_d_z_multi(
        X, n_atoms, n_times_atom, uv_constraint=uv_constraint,
        solver_d=solver_d, random_state=0, n_iter=30)

    msg = "Cost function does not go down for uv_constraint {}".format(
        uv_constraint)

    assert np.sum(np.diff(pobj) > 0) == 0, msg
    try:
        assert all([p1 >= p2 for p1, p2 in zip(pobj[:-1], pobj[1:])]), msg
    except:
        import matplotlib.pyplot as plt
        plt.semilogy(pobj - np.min(pobj) + 1e-6)
        plt.title(msg)
        plt.show()
