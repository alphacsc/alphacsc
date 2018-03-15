import pytest
import numpy as np
from numpy.testing import assert_allclose


from alphacsc.init_dict import init_uv
from alphacsc.update_d_multi import prox_uv
from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.utils import check_random_state


@pytest.mark.parametrize("uv_constraint", [
    'separate', 'joint'
])
def test_init_array(uv_constraint):
    n_trials, n_channels, n_times = 5, 3, 100
    n_times_atom, n_atoms = 10, 4

    X = np.random.randn(n_trials, n_channels, n_times)

    # Test that init_uv is doing what we expect for uv_init array
    uv = np.random.randn(n_atoms, n_channels + n_times_atom)
    uv_hat = init_uv(X, n_atoms, n_times_atom, uv_init=uv,
                     uv_constraint=uv_constraint)

    uv = prox_uv(uv, uv_constraint=uv_constraint, n_chan=n_channels)
    assert_allclose(uv_hat, uv)
    assert id(uv_hat) != id(uv)

    # Test that learn_d_z_multi is doing what we expect for uv_init array
    uv = np.random.randn(n_atoms, n_channels + n_times_atom)
    _, _, uv_hat, _ = learn_d_z_multi(
        X, n_atoms, n_times_atom, uv_init=uv, n_iter=0,
        uv_constraint=uv_constraint)

    uv = prox_uv(uv, uv_constraint=uv_constraint, n_chan=n_channels)
    assert_allclose(uv_hat, uv)


@pytest.mark.parametrize("uv_constraint", [
    'separate', 'joint'
])
def test_init_random(uv_constraint):
    """"""
    n_trials, n_channels, n_times = 5, 3, 100
    n_times_atom, n_atoms = 10, 4

    X = np.random.randn(n_trials, n_channels, n_times)

    # Test that init_uv is doing what we expect for uv_init random
    random_state = 42
    uv_hat = init_uv(X, n_atoms, n_times_atom, uv_init='random',
                     uv_constraint=uv_constraint, random_state=random_state)
    rng = check_random_state(random_state)
    uv = rng.randn(n_atoms, n_channels + n_times_atom)
    uv = prox_uv(uv, uv_constraint=uv_constraint, n_chan=n_channels)
    assert_allclose(uv_hat, uv, err_msg="The random state is not correctly "
                    "used in init_uv .")

    # Test that learn_d_z_multi is doing what we expect for uv_init random
    random_state = 27
    _, _, uv_hat, _ = learn_d_z_multi(
        X, n_atoms, n_times_atom, uv_init='random', n_iter=0,
        uv_constraint=uv_constraint, random_state=random_state)

    rng = check_random_state(random_state)
    uv = rng.randn(n_atoms, n_channels + n_times_atom)
    uv = prox_uv(uv, uv_constraint=uv_constraint, n_chan=n_channels)
    assert_allclose(uv_hat, uv, err_msg="The random state is not correctly "
                    "used in learn_d_z_multi.")


@pytest.mark.parametrize("uv_init", [
    None, 'random', 'chunk'
])
def test_init_shape(uv_init):
    """"""
    n_trials, n_channels, n_times = 5, 3, 100
    n_times_atom, n_atoms = 10, 4

    X = np.random.randn(n_trials, n_channels, n_times)

    # Test that init_uv returns correct shape
    uv_hat = init_uv(X, n_atoms, n_times_atom, uv_init=uv_init,
                     uv_constraint='separate', random_state=42)
    assert uv_hat.shape == (n_atoms, n_times_atom + n_channels)
    assert len(uv_hat.nonzero()) > 0
