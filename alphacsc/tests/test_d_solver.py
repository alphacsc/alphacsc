import pytest
from numpy.testing import assert_allclose

import numpy as np

from alphacsc.tests.conftest import N_TRIALS, N_CHANNELS, N_TIMES_ATOM, N_ATOMS

from alphacsc.loss_and_gradient import compute_objective
from alphacsc.utils import check_random_state, construct_X_multi
from alphacsc.update_d_multi import prox_d, prox_uv
from alphacsc._d_solver import get_solver_d


@pytest.fixture
def D_init(rng, shape):
    return rng.randn(*shape)


@pytest.mark.parametrize('rank1, solver_d, uv_constraint', [
    (False, 'auto', 'auto'),
    (False, 'fista', 'auto'),
    (True, 'joint', 'joint'),
    (True, 'fista', 'joint'),
    (True, 'alternate_adaptive', 'separate'),
    (True, 'alternate', 'separate'),
    (True, 'auto', 'separate')
])
@pytest.mark.parametrize('window', [True, False])
@pytest.mark.parametrize('momentum', [True, False])
def test_get_solver_d(rank1, solver_d, uv_constraint, window, momentum):
    """Tests valid values."""

    d_solver = get_solver_d(solver_d=solver_d,
                            rank1=rank1,
                            window=window,
                            momentum=momentum)

    assert d_solver is not None


@pytest.mark.parametrize('solver_d', ['alternate', 'alternate_adaptive',
                                      'joint'])
def test_get_solver_d_error_solver(solver_d):
    """Tests for the case rank1 is False and solver_d is not compatible."""

    with pytest.raises(AssertionError,
                       match="solver_d should be auto or fista. Got*"):

        get_solver_d(solver_d=solver_d,
                     rank1=False,
                     window=True,
                     momentum=False)


@pytest.mark.parametrize('solver_d', ['auto', 'fista'])
@pytest.mark.parametrize('uv_constraint', ['separate', 'joint'])
def test_get_solver_d_error_uv_constraint(solver_d, uv_constraint):
    """Tests for the case rank1 is False and uv_constraint is not
    compatible."""

    with pytest.raises(AssertionError,
                       match="If rank1 is False, uv_constraint should be*"):

        get_solver_d(solver_d=solver_d,
                     uv_constraint=uv_constraint,
                     rank1=False,
                     window=True,
                     momentum=False)


@pytest.mark.parametrize('solver_d', ['alternate', 'alternate_adaptive',
                                      'auto'])
def test_get_solver_d_error_rank1_uv_constraint(solver_d):
    """Tests for the case rank1 is True and uv_constraint is not
    compatible."""

    with pytest.raises(AssertionError,
                       match="solver_d='alternat*"):

        get_solver_d(solver_d=solver_d,
                     uv_constraint='joint',
                     rank1=True,
                     window=True,
                     momentum=False)


@pytest.mark.parametrize('solver_d', ['auto', 'fista'])
@pytest.mark.parametrize('uv_constraint', ['auto'])
@pytest.mark.parametrize('window', [True, False])
@pytest.mark.parametrize('shape', [(N_ATOMS, N_CHANNELS, N_TIMES_ATOM)])
@pytest.mark.parametrize('n_trials', [N_TRIALS])
def test_init_dictionary(X, D_init, solver_d, uv_constraint, window):
    """Tests for valid values when rank1 is False."""
    d_solver = get_solver_d(solver_d=solver_d,
                            uv_constraint=uv_constraint,
                            rank1=False,
                            window=window)

    assert d_solver is not None

    D_hat = d_solver.init_dictionary(X, N_ATOMS, N_TIMES_ATOM, D_init=D_init)

    D_init = prox_d(D_init)
    assert_allclose(D_hat, D_init)
    assert id(D_hat) != id(D_init)


@pytest.mark.parametrize('solver_d', ['joint', 'fista'])
@pytest.mark.parametrize('uv_constraint', ['joint', 'separate'])
@pytest.mark.parametrize('window', [True, False])
@pytest.mark.parametrize('shape', [(N_ATOMS, N_CHANNELS + N_TIMES_ATOM)])
@pytest.mark.parametrize('n_trials', [N_TRIALS])
def test_init_dictionary_rank1(X, D_init, solver_d, uv_constraint, window):
    """Tests for valid values when solver_d is either in 'joint' or 'fista' and
    rank1 is True."""

    d_solver = get_solver_d(solver_d=solver_d,
                            uv_constraint=uv_constraint,
                            rank1=True,
                            window=window)

    assert d_solver is not None

    D_hat = d_solver.init_dictionary(X, N_ATOMS, N_TIMES_ATOM, D_init=D_init)

    D_init = prox_uv(D_init, uv_constraint=uv_constraint,
                     n_channels=N_CHANNELS)
    assert_allclose(D_hat, D_init)
    assert id(D_hat) != id(D_init)


@pytest.mark.parametrize('solver_d', ['auto', 'alternate',
                                      'alternate_adaptive'])
@pytest.mark.parametrize('uv_constraint', ['separate'])
@pytest.mark.parametrize('window', [True, False])
@pytest.mark.parametrize('shape', [(N_ATOMS, N_CHANNELS + N_TIMES_ATOM)])
@pytest.mark.parametrize('n_trials', [N_TRIALS])
def test_init_dictionary_rank1_alternate(X, D_init, solver_d, uv_constraint,
                                         window):
    """Tests for valid values when solver_d is alternate and rank1 is True."""

    d_solver = get_solver_d(solver_d=solver_d,
                            uv_constraint=uv_constraint,
                            rank1=True,
                            window=window)

    assert d_solver is not None

    D_hat = d_solver.init_dictionary(X, N_ATOMS, N_TIMES_ATOM, D_init=D_init)

    D_init = prox_uv(D_init, uv_constraint=uv_constraint,
                     n_channels=N_CHANNELS)
    assert_allclose(D_hat, D_init)
    assert id(D_hat) != id(D_init)


@pytest.mark.parametrize('solver_d', ['joint', 'fista'])
@pytest.mark.parametrize('uv_constraint', ['joint', 'separate'])
@pytest.mark.parametrize('shape', [(N_ATOMS, N_CHANNELS + N_TIMES_ATOM)])
@pytest.mark.parametrize('n_trials', [N_TRIALS])
def test_init_dictionary_rank1_random(X, solver_d, uv_constraint,
                                      shape):
    """Tests if init_dictionary is doing what is expected from D_init random
    rank1 is True."""

    d_solver = get_solver_d(solver_d=solver_d,
                            uv_constraint=uv_constraint,
                            rank1=True,
                            window=False,
                            random_state=42)

    assert d_solver is not None

    D_hat = d_solver.init_dictionary(X, N_ATOMS, N_TIMES_ATOM, D_init='random')

    rng = check_random_state(42)

    D_init = rng.randn(*shape)

    D_init = prox_uv(D_init, uv_constraint=uv_constraint,
                     n_channels=N_CHANNELS)

    assert_allclose(D_hat, D_init)
    assert id(D_hat) != id(D_init)


@pytest.mark.parametrize('rank1, solver_d, uv_constraint, expected_shape',
                         [(True, 'alternate', 'separate',
                           (N_ATOMS, N_CHANNELS + N_TIMES_ATOM)),
                          (False, 'fista', 'auto',
                           (N_ATOMS, N_CHANNELS, N_TIMES_ATOM))
                          ])
@pytest.mark.parametrize('window', [True, False])
@pytest.mark.parametrize('n_trials', [N_TRIALS])
def test_init_dictionary_shape(X, rank1, solver_d, uv_constraint,
                               expected_shape, window):
    """Tests if the shape of dictionary complies with rank1 value."""
    d_solver = get_solver_d(solver_d=solver_d,
                            uv_constraint=uv_constraint,
                            rank1=rank1,
                            window=window)

    assert d_solver is not None

    D_hat = d_solver.init_dictionary(X, N_ATOMS, N_TIMES_ATOM)
    assert D_hat.shape == expected_shape


@pytest.mark.parametrize('rank1, solver_d, uv_constraint, shape', [
    (True, 'alternate', 'separate', (N_ATOMS, N_CHANNELS + N_TIMES_ATOM)),
    (True, 'alternate_adaptive', 'separate',
     (N_ATOMS, N_CHANNELS + N_TIMES_ATOM)),
    (True, 'joint', 'joint', (N_ATOMS, N_CHANNELS + N_TIMES_ATOM)),
    (True, 'fista', 'joint', (N_ATOMS, N_CHANNELS + N_TIMES_ATOM)),
    (False, 'fista', 'auto', (N_ATOMS, N_CHANNELS, N_TIMES_ATOM))
])
@pytest.mark.parametrize('loss', ['l2'])
@pytest.mark.parametrize('window', ['True', 'False'])
def test_update_D(rank1, solver_d, uv_constraint, window, shape,
                  z_encoder_rank1, rng):

    X = z_encoder_rank1.X
    z = z_encoder_rank1.z_hat

    def objective(uv):
        X_hat = construct_X_multi(z, D=uv, n_channels=N_CHANNELS)
        return compute_objective(X, X_hat, z_encoder_rank1.loss)

    d_solver = get_solver_d(solver_d=solver_d,
                            uv_constraint=uv_constraint,
                            rank1=rank1,
                            window=window,
                            max_iter=1000)

    uv0 = z_encoder_rank1.D_hat

    # Ensure that the known optimal point is stable
    uv = d_solver.update_D(z_encoder_rank1)
    cost = objective(uv)

    assert np.isclose(cost, 0), "optimal point not stable"
    assert np.allclose(uv, uv0), "optimal point not stable"

    uv1 = rng.normal(size=(shape))
    uv1 = prox_uv(uv1)

    cost0 = objective(uv1)

    z_encoder_rank1.D_hat = uv1

    uv = d_solver.update_D(z_encoder_rank1)
    cost1 = objective(uv)

    assert cost1 < cost0, "Learning is not going down"


@pytest.mark.parametrize('rank1, solver_d, uv_constraint, shape', [
    (True, 'alternate', 'separate', (N_ATOMS, N_CHANNELS + N_TIMES_ATOM)),
])
@pytest.mark.parametrize('loss', ['dtw', 'whitening'])
def test_update_D_error(rank1, solver_d, uv_constraint, shape,
                        z_encoder_rank1, rng):

    d_solver = get_solver_d(solver_d=solver_d,
                            uv_constraint=uv_constraint,
                            rank1=rank1,
                            max_iter=1000)

    with pytest.raises(NotImplementedError):
        d_solver.update_D(z_encoder_rank1)
