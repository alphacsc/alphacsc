import numpy as np

import pytest

from alphacsc._encoder import get_z_encoder_for
from alphacsc.init_dict import init_dictionary
from alphacsc.utils import check_random_state

N_TRIALS, N_CHANNELS, N_TIMES = 2, 3, 30
N_TIMES_ATOM, N_ATOMS = 6, 4

rng = check_random_state(42)
X = rng.randn(N_TRIALS, N_CHANNELS, N_TIMES)
X.setflags(write=False)


@pytest.fixture
def D_hat():
    return init_dictionary(X, N_ATOMS, N_TIMES_ATOM, random_state=0)


@pytest.mark.parametrize('solver_z', ['l-bfgs', 'lgcd'])
@pytest.mark.parametrize('algorithm', ['batch', 'greedy', 'online',
                                       'stochastic'])
@pytest.mark.parametrize('loss', ['l2', 'dtw', 'whitening'])
@pytest.mark.parametrize('uv_constraint', ['joint', 'separate'])
@pytest.mark.parametrize('feasible_evaluation', [True, False])
def test_get_encoder_for(solver_z, D_hat, algorithm, loss, 
                         uv_constraint, feasible_evaluation):
    """Test for valid values."""

    with get_z_encoder_for(solver=solver_z,
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           atom_support=N_TIMES_ATOM,
                           algorithm=algorithm,
                           loss=loss,
                           uv_constraint=uv_constraint,
                           feasible_evaluation=feasible_evaluation,
                           n_jobs=2,
                           use_sparse_z=False) as z_encoder:

        assert z_encoder is not None
        assert z_encoder.z_kwargs == dict()


@pytest.mark.parametrize('solver_z', [None, 'other'])
def test_get_encoder_for_error_solver_z(D_hat,  solver_z):
    """Tests for invalid values of `solver_z`."""

    with pytest.raises(ValueError) as error:
        get_z_encoder_for(solver=solver_z,
                          z_kwargs=dict(),
                          X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          n_jobs=2,
                          use_sparse_z=False)
    assert error.value.args[0] == f'unrecognized solver type: {solver_z}.'


def test_get_encoder_for_error_z_kwargs(D_hat):
    """Tests for invalid value of `z_kwargs`."""

    with pytest.raises(AssertionError) as error:
        get_z_encoder_for(solver='lgcd',
                          z_kwargs=None,
                          X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          n_jobs=2,
                          use_sparse_z=False)
    assert error.value.args[0] == 'z_kwargs should be a valid dictionary.'


def test_get_encoder_for_error_X(D_hat):
    """Tests for invalid values of `X`."""

    # test for X = None
    with pytest.raises(AssertionError) as error:
        get_z_encoder_for(solver='lgcd',
                          z_kwargs=dict(),
                          X=None,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          n_jobs=2,
                          use_sparse_z=False)
    assert error.value.args[0] == 'X should be a valid array of shape ' \
        '(n_trials, n_channels, n_times).'

    # test for invalid shape of X
    X = rng.randn(N_TRIALS, N_CHANNELS)
    with pytest.raises(AssertionError) as error:
        get_z_encoder_for(solver='lgcd',
                          z_kwargs=dict(),
                          X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          n_jobs=2,
                          use_sparse_z=False)
    assert error.value.args[0] == 'X should be a valid array of shape ' \
        '(n_trials, n_channels, n_times).'


def test_get_encoder_for_error_D_hat():
    """Tests for invalid values of `D_hat`."""

    # test for D_hat = None
    with pytest.raises(AssertionError) as error:
        get_z_encoder_for(solver='lgcd',
                          z_kwargs=dict(),
                          X=X,
                          D_hat=None,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          n_jobs=2,
                          use_sparse_z=False)
    assert error.value.args[0] == 'D_hat should be a valid array of shape '\
        '(n_trials, n_channels, n_times) or ' \
        '(n_atoms, n_channels + atom_support).'

    # test for invalid D_hat shape
    D_hat = rng.randn(N_TRIALS)
    with pytest.raises(AssertionError) as error:
        get_z_encoder_for(solver='lgcd',
                          z_kwargs=dict(),
                          X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          n_jobs=2,
                          use_sparse_z=False)
    assert error.value.args[0] == 'D_hat should be a valid array of shape '\
        '(n_trials, n_channels, n_times) or ' \
        '(n_atoms, n_channels + atom_support).'


@pytest.mark.parametrize('algorithm', [None, 'other'])
def test_get_encoder_for_error_algorithm(D_hat,  algorithm):
    """Tests for invalid values of `algorithm`."""

    with pytest.raises(AssertionError) as error:
        get_z_encoder_for(solver='lgcd',
                          z_kwargs=dict(),
                          X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          algorithm=algorithm,
                          n_jobs=2,
                          use_sparse_z=False)
    assert error.value.args[0] == f'unrecognized algorithm type: {algorithm}.'


def test_get_encoder_for_error_reg(D_hat):
    """Tests for invalid value of `reg`."""

    with pytest.raises(AssertionError) as error:
        get_z_encoder_for(solver='lgcd',
                          z_kwargs=dict(),
                          X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          reg=None,
                          n_jobs=2,
                          use_sparse_z=False)
    assert error.value.args[0] == 'reg value cannot be None.'


@pytest.mark.parametrize('loss', [None, 'other'])
def test_get_encoder_for_error_loss(D_hat,  loss):
    """Tests for invalid values of `loss`."""

    with pytest.raises(AssertionError) as error:
        get_z_encoder_for(solver='lgcd',
                          z_kwargs=dict(),
                          X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          loss=loss,
                          n_jobs=2,
                          use_sparse_z=False)
    assert error.value.args[0] == f'unrecognized loss type: {loss}.'


def test_get_encoder_for_error_loss_params(D_hat):
    """Tests for invalid value of `loss_params`."""

    with pytest.raises(AssertionError) as error:
        get_z_encoder_for(solver='lgcd',
                          z_kwargs=dict(),
                          X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          loss_params=42,
                          n_jobs=2,
                          use_sparse_z=False)
    assert error.value.args[
        0] == 'loss_params should be a valid dict or None.'


@pytest.mark.parametrize('uv_constraint', [None, 'other'])
def test_get_encoder_for_error_uv_constraint(D_hat, 
                                             uv_constraint):
    """Tests for invalid values of `uv_constraint`."""

    with pytest.raises(AssertionError) as error:
        get_z_encoder_for(solver='lgcd',
                          z_kwargs=dict(),
                          X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          uv_constraint=uv_constraint,
                          n_jobs=2,
                          use_sparse_z=False)
    assert error.value.args[0] == \
        f'unrecognized uv_constraint type: {uv_constraint}.'


def test_get_z_hat(D_hat):
    """Test for valid values."""

    # tests when use_sparse_z = False
    with get_z_encoder_for(solver='lgcd',
                           z_kwargs=dict(),
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           atom_support=N_TIMES_ATOM,
                           n_jobs=2,
                           use_sparse_z=False) as z_encoder:

        assert z_encoder is not None
        assert not z_encoder.get_z_hat().any()

        z_encoder.compute_z()
        assert z_encoder.get_z_hat().any()

    # tests when use_sparse_z = True
    with get_z_encoder_for(solver='lgcd',
                           z_kwargs=dict(),
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           atom_support=N_TIMES_ATOM,
                           n_jobs=2,
                           use_sparse_z=True) as z_encoder:

        assert z_encoder is not None

        for matrix in z_encoder.get_z_hat():
            assert not matrix.count_nonzero()

        z_encoder.compute_z()
        for matrix in z_encoder.get_z_hat():
            assert matrix.count_nonzero()


def test_get_cost(D_hat):
    """Test for valid values."""

    with get_z_encoder_for(solver='lgcd',
                           z_kwargs=dict(),
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           atom_support=N_TIMES_ATOM,
                           n_jobs=2,
                           use_sparse_z=False) as z_encoder:
        assert not z_encoder.get_z_hat().any()
        initial_cost = z_encoder.get_cost()

        z_encoder.compute_z()
        final_cost = z_encoder.get_cost()

        assert final_cost < initial_cost


def test_compute_z(D_hat):
    """Test for valid values."""

    with get_z_encoder_for(solver='lgcd',
                           z_kwargs=dict(),
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           atom_support=N_TIMES_ATOM,
                           n_jobs=2,
                           use_sparse_z=False) as z_encoder:
        z_encoder.compute_z()
        assert z_encoder.get_z_hat().any()


def test_compute_z_partial(D_hat):
    """Test for valid values."""

    with get_z_encoder_for(solver='lgcd',
                           z_kwargs=dict(),
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           atom_support=N_TIMES_ATOM,
                           n_jobs=2,
                           use_sparse_z=False) as z_encoder:

        i0 = rng.choice(N_TRIALS, 1, replace=False)
        z_encoder.compute_z_partial(i0)
        assert z_encoder.get_z_hat().any()


def test_get_sufficient_statistics(D_hat):
    """Test for valid values."""

    z_encoder = get_z_encoder_for(solver='lgcd',
                                  z_kwargs=dict(),
                                  X=X,
                                  D_hat=D_hat,
                                  n_atoms=N_ATOMS,
                                  atom_support=N_TIMES_ATOM,
                                  n_jobs=2,
                                  use_sparse_z=False)

    z_encoder.compute_z()

    ztz, ztX = z_encoder.get_sufficient_statistics()
    assert ztz is not None and ztX is not None


def test_get_sufficient_statistics_error(D_hat):
    """Test for invalid call to function."""

    z_encoder = get_z_encoder_for(solver='lgcd',
                                  z_kwargs=dict(),
                                  X=X,
                                  D_hat=D_hat,
                                  n_atoms=N_ATOMS,
                                  atom_support=N_TIMES_ATOM,
                                  n_jobs=2,
                                  use_sparse_z=False)

    # test before calling compute_z
    with pytest.raises(AssertionError) as error:
        z_encoder.get_sufficient_statistics()

    assert error.value.args[0] == \
        'compute_z should be called to access the statistics.'


def test_get_sufficient_statistics_partial(D_hat):
    """Test for valid values."""

    z_encoder = get_z_encoder_for(solver='lgcd',
                                  z_kwargs=dict(),
                                  X=X,
                                  D_hat=D_hat,
                                  n_atoms=N_ATOMS,
                                  atom_support=N_TIMES_ATOM,
                                  n_jobs=2,
                                  use_sparse_z=False)

    i0 = rng.choice(N_TRIALS, 1, replace=False)
    z_encoder.compute_z_partial(i0)

    ztz_i0, ztX_i0 = z_encoder.get_sufficient_statistics_partial()
    assert ztz_i0 is not None and ztX_i0 is not None


def test_get_sufficient_statistics_partial_error(D_hat):
    """Test for invalid call to function."""

    z_encoder = get_z_encoder_for(solver='lgcd',
                                  z_kwargs=dict(),
                                  X=X,
                                  D_hat=D_hat,
                                  n_atoms=N_ATOMS,
                                  atom_support=N_TIMES_ATOM,
                                  n_jobs=2,
                                  use_sparse_z=False)

    # test before calling compute_z_partial
    with pytest.raises(AssertionError) as error:
        z_encoder.get_sufficient_statistics_partial()

    assert error.value.args[0] == \
        'compute_z_partial should be called to access the statistics.'


def test_add_one_atom(D_hat):
    """Test for valid values."""

    with get_z_encoder_for(solver='lgcd',
                           z_kwargs=dict(),
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           atom_support=N_TIMES_ATOM,
                           n_jobs=2,
                           use_sparse_z=False) as z_encoder:
        new_atom = np.random.rand(N_CHANNELS + N_TIMES_ATOM)
        z_encoder.add_one_atom(new_atom)
        n_atoms_plus_one = z_encoder.D_hat.shape[0]
        assert n_atoms_plus_one == N_ATOMS + 1
