import pytest
import numpy as np

from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.utils import check_random_state


@pytest.mark.parametrize('loss', ['l2', 'dtw', 'whitening'])
@pytest.mark.parametrize('solver_d, uv_constraint', [
    ('joint', 'joint'), ('joint', 'separate'),
    # ('alternate', 'separate'), ('l-bfgs', 'box'),
    ('alternate_adaptive', 'separate')
])
def test_learn_d_z_multi(loss, solver_d, uv_constraint):
    # smoke test for learn_d_z_multi
    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4

    loss_params = dict(gamma=1, sakoe_chiba_band=10, ordar=10)

    rng = check_random_state(42)
    X = rng.randn(n_trials, n_channels, n_times)
    pobj, times, uv_hat, z_hat = learn_d_z_multi(
        X, n_atoms, n_times_atom, uv_constraint=uv_constraint,
        solver_d=solver_d, random_state=0, n_iter=30,
        solver_z='l-bfgs',
        loss=loss, loss_params=loss_params)

    msg = "Cost function does not go down for uv_constraint {}".format(
        uv_constraint)

    try:
        assert np.sum(np.diff(pobj) > 0) == 0, msg
    except AssertionError:
        import matplotlib.pyplot as plt
        plt.semilogy(pobj - np.min(pobj) + 1e-6)
        plt.title(msg)
        plt.show()
        raise
