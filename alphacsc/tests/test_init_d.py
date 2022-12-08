import pytest
import numpy as np
import functools
from numpy.testing import assert_allclose


from alphacsc.init_dict import init_dictionary
from alphacsc.update_d_multi import prox_uv, prox_d
from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.utils.validation import check_random_state

from alphacsc.tests.conftest import parametrize_solver_and_constraint


@parametrize_solver_and_constraint
def test_init_array(rank1, solver_d, uv_constraint):
    n_trials, n_channels, n_times = 5, 3, 100
    n_times_atom, n_atoms = 10, 4

    if rank1:
        expected_shape = (n_atoms, n_channels + n_times_atom)
        prox = functools.partial(prox_uv, uv_constraint=uv_constraint,
                                 n_channels=n_channels)
    else:
        expected_shape = (n_atoms, n_channels, n_times_atom)
        prox = prox_d

    X = np.random.randn(n_trials, n_channels, n_times)

    # Test that init_dictionary is doing what we expect for D_init array
    D_init = np.random.randn(*expected_shape)
    D_hat = init_dictionary(X, n_atoms, n_times_atom, D_init=D_init,
                            rank1=rank1, uv_constraint=uv_constraint)

    D_init = prox(D_init)
    assert_allclose(D_hat, D_init)
    assert id(D_hat) != id(D_init)

    # Test that learn_d_z_multi is doing what we expect for D_init array
    D_init = np.random.randn(*expected_shape)
    _, _, D_hat, _, _ = learn_d_z_multi(
        X, n_atoms, n_times_atom, D_init=D_init, n_iter=0,
        rank1=rank1, solver_d=solver_d, uv_constraint=uv_constraint
    )

    D_init = prox(D_init)
    assert_allclose(D_hat, D_init)


@parametrize_solver_and_constraint
def test_init_random(rank1, solver_d, uv_constraint):
    """"""
    n_trials, n_channels, n_times = 5, 3, 100
    n_times_atom, n_atoms = 10, 4

    if rank1:
        expected_shape = (n_atoms, n_channels + n_times_atom)
        prox = functools.partial(prox_uv, uv_constraint=uv_constraint,
                                 n_channels=n_channels)
    else:
        expected_shape = (n_atoms, n_channels, n_times_atom)
        prox = prox_d

    X = np.random.randn(n_trials, n_channels, n_times)

    # Test that init_dictionary is doing what we expect for D_init random
    random_state = 42
    D_hat = init_dictionary(X, n_atoms, n_times_atom, D_init='random',
                            rank1=rank1, uv_constraint=uv_constraint,
                            random_state=random_state)
    rng = check_random_state(random_state)

    D_init = rng.randn(*expected_shape)
    D_init = prox(D_init)
    assert_allclose(D_hat, D_init, err_msg="The random state is not correctly "
                    "used in init_dictionary .")

    # Test that learn_d_z_multi is doing what we expect for D_init random
    random_state = 27
    _, _, D_hat, _, _ = learn_d_z_multi(
        X, n_atoms, n_times_atom, D_init='random', n_iter=0,
        rank1=rank1, solver_d=solver_d, uv_constraint=uv_constraint,
        random_state=random_state
    )

    rng = check_random_state(random_state)
    D_init = rng.randn(*expected_shape)
    D_init = prox(D_init)
    assert_allclose(D_hat, D_init, err_msg="The random state is not correctly "
                    "used in learn_d_z_multi.")


@pytest.mark.parametrize("rank1", [True, False])
@pytest.mark.parametrize("D_init", [
    None, 'random', 'chunk',
])
def test_init_shape(D_init, rank1):
    n_trials, n_channels, n_times = 5, 3, 100
    n_times_atom, n_atoms = 10, 4

    X = np.random.randn(n_trials, n_channels, n_times)

    expected_shape = (n_atoms, n_channels, n_times_atom)
    if rank1:
        expected_shape = (n_atoms, n_channels + n_times_atom)

    # Test that init_dictionary returns correct shape
    uv_hat = init_dictionary(X, n_atoms, n_times_atom, D_init=D_init,
                             rank1=rank1, uv_constraint='separate',
                             random_state=42)
    assert uv_hat.shape == expected_shape
