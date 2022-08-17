import pytest
from numpy.testing import assert_allclose

import numpy as np

from alphacsc.tests.conftest import N_CHANNELS, N_TIMES_ATOM, N_ATOMS

from alphacsc.update_d_multi import prox_d, prox_uv
from alphacsc._d_solver import get_solver_d, check_solver_and_constraints
from alphacsc._z_encoder import get_z_encoder_for


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

    d_solver = get_solver_d(N_CHANNELS,
                            N_ATOMS,
                            N_TIMES_ATOM,
                            solver_d=solver_d,
                            rank1=rank1,
                            D_init='random',
                            window=window,
                            momentum=momentum)

    assert d_solver is not None


@pytest.mark.parametrize('solver_d', ['alternate', 'alternate_adaptive',
                                      'joint'])
def test_get_solver_d_error_solver(solver_d):
    """Tests for the case rank1 is False and solver_d is not compatible."""

    with pytest.raises(AssertionError,
                       match="solver_d should be auto or fista. Got*"):

        get_solver_d(N_CHANNELS,
                     N_ATOMS,
                     N_TIMES_ATOM,
                     solver_d=solver_d,
                     rank1=False,
                     D_init='random',
                     window=True,
                     momentum=False)


@pytest.mark.parametrize('solver_d', ['auto', 'fista'])
@pytest.mark.parametrize('uv_constraint', ['separate', 'joint'])
def test_get_solver_d_error_uv_constraint(solver_d, uv_constraint):
    """Tests for the case rank1 is False and uv_constraint is not
    compatible."""

    with pytest.raises(AssertionError,
                       match="If rank1 is False, uv_constraint should be*"):

        get_solver_d(N_CHANNELS,
                     N_ATOMS,
                     N_TIMES_ATOM,
                     solver_d=solver_d,
                     uv_constraint=uv_constraint,
                     rank1=False,
                     D_init='random',
                     window=True,
                     momentum=False)


@pytest.mark.parametrize('solver_d', ['alternate', 'alternate_adaptive',
                                      'auto'])
def test_get_solver_d_error_rank1_uv_constraint(solver_d):
    """Tests for the case rank1 is True and uv_constraint is not
    compatible."""

    with pytest.raises(AssertionError,
                       match="solver_d='alternat*"):

        get_solver_d(N_CHANNELS,
                     N_ATOMS,
                     N_TIMES_ATOM,
                     solver_d=solver_d,
                     uv_constraint='joint',
                     rank1=True,
                     D_init='random',
                     window=True,
                     momentum=False)


@pytest.mark.parametrize('solver_d, uv_constraint, rank1, shape', [
    ('auto', 'auto', False, (N_ATOMS, N_CHANNELS, N_TIMES_ATOM)),
    ('fista', 'auto', False, (N_ATOMS, N_CHANNELS, N_TIMES_ATOM)),
    ('joint', 'joint', True, (N_ATOMS, N_CHANNELS+N_TIMES_ATOM)),
    ('joint', 'separate', True, (N_ATOMS, N_CHANNELS+N_TIMES_ATOM)),
    ('fista', 'joint', True, (N_ATOMS, N_CHANNELS+N_TIMES_ATOM)),
    ('fista', 'separate', True, (N_ATOMS, N_CHANNELS+N_TIMES_ATOM)),
    ('auto', 'separate', True, (N_ATOMS, N_CHANNELS+N_TIMES_ATOM)),
    ('alternate', 'separate', True, (N_ATOMS, N_CHANNELS+N_TIMES_ATOM)),
    ('alternate_adaptive', 'separate', True, (N_ATOMS, N_CHANNELS+N_TIMES_ATOM)),  # noqa
])
@pytest.mark.parametrize('window', [True, False])
@pytest.mark.parametrize('D_init', ['random', 'chunk'])
def test_init_dictionary(X, D_init, solver_d, uv_constraint, rank1, shape,
                         window):
    """Tests for valid values when D_init is specified as type."""
    d_solver = get_solver_d(N_CHANNELS,
                            N_ATOMS,
                            N_TIMES_ATOM,
                            solver_d=solver_d,
                            uv_constraint=uv_constraint,
                            D_init=D_init,
                            rank1=rank1,
                            window=window)

    assert d_solver is not None

    D_hat = d_solver.init_dictionary(X)

    assert D_hat is not None
    assert D_hat.shape == shape


@pytest.mark.parametrize('solver_d', ['auto', 'fista'])
@pytest.mark.parametrize('uv_constraint', ['auto'])
@pytest.mark.parametrize('shape', [(N_ATOMS, N_CHANNELS, N_TIMES_ATOM)])
@pytest.mark.parametrize('window', [True, False])
def test_init_dictionary_initial_D_init(X, D_init, solver_d, window,
                                        uv_constraint, shape):
    """Tests if init_dictionary is doing what is expected when rank1 is False
    and initial D_init is provided.
    """

    d_solver = get_solver_d(N_CHANNELS,
                            N_ATOMS,
                            N_TIMES_ATOM,
                            solver_d=solver_d,
                            uv_constraint=uv_constraint,
                            rank1=False,
                            window=window,
                            D_init=D_init,
                            random_state=42)

    assert d_solver is not None

    D_hat = d_solver.init_dictionary(X)

    assert D_hat is not None

    assert D_hat.shape == shape

    D_init = prox_d(D_init)
    assert_allclose(D_hat, D_init)
    assert id(D_hat) != id(D_init)


@pytest.mark.parametrize('solver_d', ['joint', 'fista'])
@pytest.mark.parametrize('uv_constraint', ['joint', 'separate'])
@pytest.mark.parametrize('shape', [(N_ATOMS, N_CHANNELS + N_TIMES_ATOM)])
def test_init_dictionary_rank1_initial_D_init(X, D_init, solver_d,
                                              uv_constraint, shape):
    """Tests if init_dictionary is doing what is expected  when rank1=True and
    initial D_init is provided."""

    d_solver = get_solver_d(N_CHANNELS,
                            N_ATOMS,
                            N_TIMES_ATOM,
                            solver_d=solver_d,
                            uv_constraint=uv_constraint,
                            rank1=True,
                            window=False,
                            D_init=D_init,
                            random_state=42)

    assert d_solver is not None

    D_hat = d_solver.init_dictionary(X)

    assert D_hat is not None

    assert D_hat.shape == (N_ATOMS, N_CHANNELS + N_TIMES_ATOM)

    D_init = prox_uv(D_init, uv_constraint=uv_constraint,
                     n_channels=N_CHANNELS)
    assert_allclose(D_hat, D_init)
    assert id(D_hat) != id(D_init)


@pytest.mark.parametrize('rank1, solver_d, uv_constraint', [
    (True, 'alternate', 'separate'),
    (True, 'alternate_adaptive', 'separate'),
    (True, 'joint', 'joint'),
    (True, 'fista', 'joint'),
    (False, 'fista', 'auto'),
])
@pytest.mark.parametrize('window', ['True', 'False'])
def test_update_D(D_hat, rank1, solver_d, uv_constraint, window,
                  z_encoder):

    d_solver = get_solver_d(N_CHANNELS,
                            N_ATOMS,
                            N_TIMES_ATOM,
                            solver_d=solver_d,
                            uv_constraint=uv_constraint,
                            D_init=D_hat,
                            rank1=rank1,
                            window=window,
                            max_iter=1000)

    d_hat0 = d_solver.init_dictionary(z_encoder.X)

    # Ensure that the known optimal point is stable
    d_hat = d_solver.update_D(z_encoder)
    cost = z_encoder.compute_objective(d_hat)

    assert np.isclose(cost, 0), "optimal point not stable"
    assert np.allclose(d_hat, d_hat0), "optimal point not stable"

    # ----------------

    d_solver = get_solver_d(N_CHANNELS,
                            N_ATOMS,
                            N_TIMES_ATOM,
                            solver_d=solver_d,
                            uv_constraint=uv_constraint,
                            D_init='random',
                            rank1=rank1,
                            window=window,
                            max_iter=1000)

    d_hat1 = d_solver.init_dictionary(z_encoder.X)
    assert not np.allclose(d_hat0, d_hat1)

    cost1 = z_encoder.compute_objective(d_hat1)

    d_hat2 = d_solver.update_D(z_encoder)
    cost2 = z_encoder.compute_objective(d_hat2)

    assert cost2 < cost1, "Learning is not going down"


@pytest.mark.parametrize('rank1, solver_d, uv_constraint', [
    (False, 'fista', 'auto'),
    (True, 'joint', 'joint'),
    (True, 'fista', 'joint'),
    (True, 'alternate_adaptive', 'separate'),
    (True, 'alternate', 'separate'),
])
@pytest.mark.parametrize('window', [True, False])
def test_add_one_atom(X, rank1, solver_d, uv_constraint, window):
    """Tests valid values."""

    d_solver = get_solver_d(N_CHANNELS,
                            N_ATOMS,
                            N_TIMES_ATOM,
                            solver_d=solver_d,
                            rank1=rank1,
                            D_init='greedy',
                            window=window)

    D_hat = d_solver.init_dictionary(X)

    with get_z_encoder_for(X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           n_times_atom=N_TIMES_ATOM,
                           n_jobs=2) as z_encoder:

        n_atoms_initial = d_solver.D_hat.shape[0]
        assert n_atoms_initial == 0
        d_solver.add_one_atom(z_encoder)
        n_atoms_plus_one = d_solver.D_hat.shape[0]
        assert n_atoms_plus_one == n_atoms_initial + 1

        d_solver.add_one_atom(z_encoder)
        n_atoms_plus_two = d_solver.D_hat.shape[0]

        assert n_atoms_plus_two == n_atoms_initial + 2
