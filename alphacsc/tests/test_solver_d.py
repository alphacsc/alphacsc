import pytest
from numpy.testing import assert_allclose
from alphacsc.tests.conftest import parametrize_solver_and_constraint

from alphacsc._solver_d import check_solver_and_constraints, get_solver_d

from alphacsc.update_d_multi import prox_d, prox_uv
from alphacsc.utils import check_random_state

N_TRIALS, N_CHANNELS, N_TIMES = 5, 3, 100
N_TIMES_ATOM, N_ATOMS = 10, 4


@pytest.fixture
def rng():
    return check_random_state(42)


@pytest.fixture
def X(rng, n_trials):
    return rng.randn(n_trials, N_CHANNELS, N_TIMES)


@pytest.fixture
def D_init(rng, shape):
    return rng.randn(*shape)


@pytest.mark.parametrize('solver_d', ['auto', 'fista'])
@pytest.mark.parametrize('uv_constraint', ['auto'])
def test_check_solver_and_constraints(solver_d, uv_constraint):
    """Tests for valid values when rank1 is False."""

    solver_d_, uv_constraint_ = check_solver_and_constraints(False, solver_d,
                                                             uv_constraint)

    assert solver_d_ == 'fista'
    assert uv_constraint_ == 'auto'


@pytest.mark.parametrize('solver_d', ['auto', 'fista'])
@pytest.mark.parametrize('uv_constraint', ['joint', 'separate'])
def test_check_solver_and_constraints_error(solver_d, uv_constraint):
    """Tests for the case rank1 is False and params are not compatible."""

    with pytest.raises(AssertionError,
                       match="If rank1 is False, uv_constraint should be*"):

        check_solver_and_constraints(False, solver_d, uv_constraint)


@pytest.mark.parametrize('solver_d', ['auto', 'alternate',
                                      'alternate_adaptive'])
@pytest.mark.parametrize('uv_constraint', ['auto', 'separate'])
def test_check_solver_and_constraints_rank1_alternate(solver_d, uv_constraint):
    """Tests for valid values when solver is alternate and rank1 is True."""

    solver_d_, uv_constraint_ = check_solver_and_constraints(True, solver_d,
                                                             uv_constraint)

    if solver_d == 'auto':
        solver_d = 'alternate_adaptive'

    assert solver_d_ == solver_d
    assert uv_constraint_ == 'separate'


@pytest.mark.parametrize('solver_d', ['joint', 'fista'])
@pytest.mark.parametrize('uv_constraint', ['auto', 'joint', 'separate'])
def test_check_solver_and_constraints_rank1(solver_d, uv_constraint):
    """Tests for valid values when solver_d is either in 'joint' or 'fista' and
    rank1 is True."""

    solver_d_, uv_constraint_ = check_solver_and_constraints(True, solver_d,
                                                             uv_constraint)

    if uv_constraint == 'auto':
        uv_constraint = 'joint'

    assert solver_d_ == solver_d
    assert uv_constraint_ == uv_constraint


@pytest.mark.parametrize('solver_d', ['auto', 'alternate',
                                      'alternate_adaptive'])
@pytest.mark.parametrize('uv_constraint', ['joint'])
def test_check_solver_and_constraints_rank1_error(solver_d, uv_constraint):
    """Tests for the case when rank1 is True and params are not compatible.
    """
    with pytest.raises(AssertionError,
                       match="solver_d=*"):

        check_solver_and_constraints(True, solver_d, uv_constraint)


@parametrize_solver_and_constraint
@pytest.mark.parametrize('window', [True, False])
@pytest.mark.parametrize('momentum', [True, False])
def test_get_solver_d(rank1, solver_d, uv_constraint, window, momentum):
    """Tests valid values."""

    d_solver = get_solver_d(solver_d=solver_d,
                            rank1=rank1,
                            uv_constraint=uv_constraint,
                            window=window,
                            momentum=momentum)

    assert d_solver is not None


@pytest.mark.parametrize('solver_d', ['auto', 'fista'])
@pytest.mark.parametrize('uv_constraint', ['joint', 'separate'])
@pytest.mark.parametrize('window', [True, False])
@pytest.mark.parametrize('momentum', [True, False])
def test_get_solver_d_error(solver_d, uv_constraint, window, momentum):
    """Tests for the case rank1 is False and params are not compatible."""

    with pytest.raises(AssertionError,
                       match="If rank1 is False, uv_constraint should be*"):

        get_solver_d(solver_d=solver_d,
                     rank1=False,
                     uv_constraint=uv_constraint,
                     window=window,
                     momentum=momentum)


@pytest.mark.parametrize('solver_d', ['auto', 'alternate',
                                      'alternate_adaptive'])
@pytest.mark.parametrize('uv_constraint', ['joint'])
@pytest.mark.parametrize('window', [True, False])
@pytest.mark.parametrize('momentum', [True, False])
def test_get_solver_d_rank1_error(solver_d, uv_constraint, window, momentum):
    """Tests for the case when rank1 is True and params are not compatible.
    """
    with pytest.raises(AssertionError,
                       match="solver_d=*"):

        get_solver_d(solver_d=solver_d,
                     rank1=False,
                     uv_constraint=uv_constraint,
                     window=window,
                     momentum=momentum)


@pytest.mark.parametrize('solver_d', ['auto', 'fista'])
@pytest.mark.parametrize('uv_constraint', ['auto'])
@pytest.mark.parametrize('window', [True, False])
@pytest.mark.parametrize('shape', [(N_ATOMS, N_CHANNELS, N_TIMES_ATOM)])
@pytest.mark.parametrize('n_trials', [N_TRIALS])
def test_init_dictionary(X, D_init, solver_d, uv_constraint, window):
    """Tests for valid values when rank1 is False."""
    d_solver = get_solver_d(solver_d=solver_d,
                            rank1=False,
                            uv_constraint=uv_constraint,
                            window=window)

    assert d_solver is not None

    D_hat = d_solver.init_dictionary(X, N_ATOMS, N_TIMES_ATOM, D_init=D_init)

    D_init = prox_d(D_init)
    assert_allclose(D_hat, D_init)
    assert id(D_hat) != id(D_init)


@pytest.mark.parametrize('solver_d', ['auto', 'alternate',
                                      'alternate_adaptive'])
@pytest.mark.parametrize('uv_constraint', ['auto', 'separate'])
@pytest.mark.parametrize('window', [True, False])
@pytest.mark.parametrize('shape', [(N_ATOMS, N_CHANNELS + N_TIMES_ATOM)])
@pytest.mark.parametrize('n_trials', [N_TRIALS])
def test_init_dictionary_rank1_alternate(X, D_init, solver_d, uv_constraint,
                                         window):
    """Tests for valid values when solver_d is alternate and rank1 is True."""

    d_solver = get_solver_d(solver_d=solver_d,
                            rank1=True,
                            uv_constraint=uv_constraint,
                            window=window)

    assert d_solver is not None

    D_hat = d_solver.init_dictionary(X, N_ATOMS, N_TIMES_ATOM, D_init=D_init)

    D_init = prox_uv(D_init, uv_constraint=d_solver.uv_constraint,
                     n_channels=N_CHANNELS)
    assert_allclose(D_hat, D_init)
    assert id(D_hat) != id(D_init)


@pytest.mark.parametrize('solver_d', ['joint', 'fista'])
@pytest.mark.parametrize('uv_constraint', ['auto', 'joint', 'separate'])
@pytest.mark.parametrize('window', [True, False])
@pytest.mark.parametrize('shape', [(N_ATOMS, N_CHANNELS + N_TIMES_ATOM)])
@pytest.mark.parametrize('n_trials', [N_TRIALS])
def test_init_dictionary_rank1(X, D_init, solver_d, uv_constraint, window):
    """Tests for valid values when solver_d is either in 'joint' or 'fista' and
    rank1 is True."""

    d_solver = get_solver_d(solver_d=solver_d,
                            rank1=True,
                            uv_constraint=uv_constraint,
                            window=window)

    assert d_solver is not None

    D_hat = d_solver.init_dictionary(X, N_ATOMS, N_TIMES_ATOM, D_init=D_init)

    D_init = prox_uv(D_init, uv_constraint=d_solver.uv_constraint,
                     n_channels=N_CHANNELS)
    assert_allclose(D_hat, D_init)
    assert id(D_hat) != id(D_init)


@pytest.mark.parametrize('solver_d', ['joint', 'fista'])
@pytest.mark.parametrize('uv_constraint', ['auto', 'joint', 'separate'])
@pytest.mark.parametrize('shape', [(N_ATOMS, N_CHANNELS + N_TIMES_ATOM)])
@pytest.mark.parametrize('n_trials', [N_TRIALS])
def test_init_dictionary_rank1_random(X, solver_d, uv_constraint,
                                      shape):
    """Tests is init_dictionary is doing what is expected from D_init random
    rank1 is True."""

    d_solver = get_solver_d(solver_d=solver_d,
                            rank1=True,
                            uv_constraint=uv_constraint,
                            window=False,
                            random_state=42)

    assert d_solver is not None

    D_hat = d_solver.init_dictionary(X, N_ATOMS, N_TIMES_ATOM, D_init='random')

    rng = check_random_state(42)

    D_init = rng.randn(*shape)

    D_init = prox_uv(D_init, uv_constraint=d_solver.uv_constraint,
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
def test_init_shape(X, rank1, solver_d, uv_constraint, expected_shape, window):
    """Tests if the shape of dictionary complies with rank1 value."""
    d_solver = get_solver_d(solver_d=solver_d,
                            rank1=rank1,
                            uv_constraint=uv_constraint,
                            window=window)

    assert d_solver is not None

    D_hat = d_solver.init_dictionary(X, N_ATOMS, N_TIMES_ATOM)
    assert D_hat.shape == expected_shape
