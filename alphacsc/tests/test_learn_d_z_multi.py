import pytest
import numpy as np

from alphacsc.learn_d_z_multi import learn_d_z_multi


@pytest.mark.parametrize('uv_constraint', ['joint', 'separate'])
def test_learn_d_z_multi(uv_constraint):
    # smoke test for learn_d_z_multi
    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4

    X = np.random.randn(n_trials, n_channels, n_times)
    pobj, times, uv_hat, Z_hat = learn_d_z_multi(
        X, n_atoms, n_times_atom, uv_constraint=uv_constraint,
        random_state=0, n_iter=20)

    msg = "Cost function does not go down for uv_constraint {}".format(
        uv_constraint)
    assert all([p1 > p2 for p1, p2 in zip(pobj[:-1], pobj[1:])]), msg
