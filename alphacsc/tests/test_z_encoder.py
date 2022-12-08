import numpy as np

import pytest

from alphacsc._z_encoder import get_z_encoder_for
from alphacsc.loss_and_gradient import compute_objective
from alphacsc.utils.convolution import construct_X_multi
from alphacsc.utils.compute_constants import compute_ztz, compute_ztX

from .conftest import N_ATOMS, N_TIMES_ATOM, N_CHANNELS


@pytest.fixture
def requires_dicodile(solver):
    if solver == 'dicodile':
        return pytest.importorskip('dicodile')


@pytest.mark.parametrize('solver', ['l-bfgs', 'lgcd'])
@pytest.mark.parametrize('n_trials', [1, 2, 5])
@pytest.mark.parametrize('rank1', [True, False])
def test_get_encoder_for_alphacsc(X, solver, D_hat):
    """Test for valid values for alphacsc backend."""

    with get_z_encoder_for(solver=solver,
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           n_times_atom=N_TIMES_ATOM,
                           n_jobs=2) as z_encoder:

        assert z_encoder is not None


@pytest.mark.parametrize('solver, n_trials, rank1', [('dicodile', 1, False),
                                                     ('dicodile', 1, True)])
def test_get_encoder_for_dicodile(X, D_hat, solver, requires_dicodile):
    """Test for valid values for dicodile backend."""

    with get_z_encoder_for(solver=solver,
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           n_times_atom=N_TIMES_ATOM,
                           n_jobs=2) as z_encoder:

        assert z_encoder is not None


@pytest.mark.parametrize('solver, n_trials, rank1', [('dicodile', 2, False)])
def test_get_encoder_for_dicodile_error_n_trials(solver, X, D_hat,
                                                 requires_dicodile):
    """Test for invalid n_trials value for dicodile backend."""

    with pytest.raises(AssertionError,
                       match="X should be a valid array of shape*"):
        get_z_encoder_for(solver=solver,
                          X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          n_times_atom=N_TIMES_ATOM,
                          n_jobs=2)


@pytest.mark.parametrize('rank1', [True])
@pytest.mark.parametrize('solver', [None, 'other'])
def test_get_encoder_for_error_solver(X, D_hat,  solver):
    """Tests for invalid values of `solver`."""

    with pytest.raises(ValueError,
                       match=f"unrecognized solver type: {solver}."):
        get_z_encoder_for(solver=solver,
                          X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          n_times_atom=N_TIMES_ATOM,
                          n_jobs=2)


@pytest.mark.parametrize('rank1', [True])
def test_get_encoder_for_error_solver_kwargs(X, D_hat):
    """Tests for invalid value of `solver_kwargs`."""

    with pytest.raises(AssertionError, match=".*solver_kwargs should.*"):
        get_z_encoder_for(solver_kwargs=None,
                          X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          n_times_atom=N_TIMES_ATOM,
                          n_jobs=2)


@pytest.mark.parametrize('rank1', [True])
@pytest.mark.parametrize('X_error', [None, np.zeros([2, N_CHANNELS])])
def test_get_encoder_for_error_X(X_error, D_hat):
    """Tests for invalid values of `X`."""

    with pytest.raises(AssertionError,
                       match="X should be a valid array of shape.*"):
        get_z_encoder_for(X=X_error,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          n_times_atom=N_TIMES_ATOM,
                          n_jobs=2)


@pytest.mark.parametrize('D_init', [None, np.zeros(2)])
def test_get_encoder_for_error_D_hat(X, D_init):
    """Tests for invalid values of `D_hat`."""

    with pytest.raises(AssertionError,
                       match="D_hat should be a valid array of shape.*"):
        get_z_encoder_for(X=X,
                          D_hat=D_init,
                          n_atoms=N_ATOMS,
                          n_times_atom=N_TIMES_ATOM,
                          n_jobs=2)


@pytest.mark.parametrize('rank1', [True])
def test_get_encoder_for_error_reg(X, D_hat):
    """Tests for invalid value of `reg`."""

    with pytest.raises(AssertionError,
                       match="reg value cannot be None."):
        get_z_encoder_for(X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          n_times_atom=N_TIMES_ATOM,
                          reg=None,
                          n_jobs=2)


@pytest.mark.parametrize('solver, n_trials, rank1',
                         [('l-bfgs', 3, True),
                          ('dicodile', 1, False)
                          ])
def test_get_z_hat(solver, X, D_hat, requires_dicodile):
    """Test for valid values."""

    with get_z_encoder_for(solver=solver,
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           n_times_atom=N_TIMES_ATOM,
                           n_jobs=2) as z_encoder:

        assert z_encoder is not None
        assert not z_encoder.get_z_hat().any()

        z_encoder.compute_z()
        assert z_encoder.get_z_hat().any()


@pytest.mark.parametrize('solver, n_trials, rank1',
                         [('l-bfgs', 3, True),
                          ('dicodile', 1, False)])
def test_get_cost(solver, X, D_hat, requires_dicodile):
    """Test for valid values."""

    with get_z_encoder_for(solver=solver,
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           n_times_atom=N_TIMES_ATOM,
                           n_jobs=2) as z_encoder:
        initial_cost = z_encoder.get_cost()

        z_encoder.compute_z()
        z_hat = z_encoder.get_z_hat()
        final_cost = z_encoder.get_cost()

        assert final_cost < initial_cost

        X_hat = construct_X_multi(z_hat, D_hat, n_channels=N_CHANNELS)
        cost = compute_objective(X=X, X_hat=X_hat, z_hat=z_hat, reg=0.1,
                                 D=D_hat)

        assert np.isclose(cost, final_cost)


@pytest.mark.parametrize('solver, n_trials, rank1', [('lgcd', 2, True),
                                                     ('l-bfgs', 5, False),
                                                     ('dicodile', 1, False)])
def test_compute_z(solver, X, D_hat, requires_dicodile):
    """Test for valid values."""

    with get_z_encoder_for(solver=solver,
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           n_times_atom=N_TIMES_ATOM,
                           n_jobs=2) as z_encoder:
        z_encoder.compute_z()
        assert z_encoder.get_z_hat().any()


@pytest.mark.parametrize('rank1', [True])
def test_compute_z_partial(X, D_hat, n_trials, rng):
    """Test for valid values."""

    with get_z_encoder_for(X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           n_times_atom=N_TIMES_ATOM,
                           n_jobs=2) as z_encoder:

        i0 = rng.choice(n_trials, 1, replace=False)
        z_encoder.compute_z_partial(i0)
        assert z_encoder.get_z_hat().any()


@pytest.mark.parametrize('solver, n_trials, rank1', [('lgcd', 2, True),
                                                     ('l-bfgs', 5, False),
                                                     ('dicodile', 1, False)])
def test_get_sufficient_statistics(solver, X, D_hat, requires_dicodile):
    """Test for valid values."""

    z_encoder = get_z_encoder_for(solver=solver,
                                  X=X,
                                  D_hat=D_hat,
                                  n_atoms=N_ATOMS,
                                  n_times_atom=N_TIMES_ATOM,
                                  n_jobs=2)

    z_encoder.compute_z()
    z_hat = z_encoder.get_z_hat()

    ztz, ztX = z_encoder.get_sufficient_statistics()
    assert ztz is not None and np.allclose(ztz, compute_ztz(z_hat,
                                                            N_TIMES_ATOM))

    assert ztX is not None and np.allclose(ztX, compute_ztX(z_hat, X))


@pytest.mark.parametrize('solver, n_trials, rank1', [('lgcd', 2, True),
                                                     ('l-bfgs', 5, False),
                                                     ('dicodile', 1, False)
                                                     ])
def test_get_sufficient_statistics_error(solver, X, D_hat,
                                         requires_dicodile):
    """Test for invalid call to function."""

    z_encoder = get_z_encoder_for(solver=solver,
                                  X=X,
                                  D_hat=D_hat,
                                  n_atoms=N_ATOMS,
                                  n_times_atom=N_TIMES_ATOM,
                                  n_jobs=2)

    # test before calling compute_z
    with pytest.raises(AssertionError,
                       match="compute_z should be called.*"):
        z_encoder.get_sufficient_statistics()


@pytest.mark.parametrize('rank1', [True])
def test_get_sufficient_statistics_partial(X, D_hat, n_trials, rng):
    """Test for valid values."""

    z_encoder = get_z_encoder_for(X=X,
                                  D_hat=D_hat,
                                  n_atoms=N_ATOMS,
                                  n_times_atom=N_TIMES_ATOM,
                                  n_jobs=2)

    i0 = rng.choice(n_trials, 1, replace=False)
    z_encoder.compute_z_partial(i0)

    ztz_i0, ztX_i0 = z_encoder.get_sufficient_statistics_partial()
    assert ztz_i0 is not None and ztX_i0 is not None


@pytest.mark.parametrize('rank1', [True])
def test_get_sufficient_statistics_partial_error(X, D_hat):
    """Test for invalid call to function."""

    z_encoder = get_z_encoder_for(X=X,
                                  D_hat=D_hat,
                                  n_atoms=N_ATOMS,
                                  n_times_atom=N_TIMES_ATOM,
                                  n_jobs=2)

    # test before calling compute_z_partial
    with pytest.raises(AssertionError,
                       match="compute_z_partial should be called.*"):
        z_encoder.get_sufficient_statistics_partial()
