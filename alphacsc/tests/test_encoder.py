import numpy as np

import pytest

from alphacsc._encoder import get_z_encoder_for
from alphacsc.init_dict import init_dictionary
from alphacsc.utils import check_random_state

N_TRIALS, N_CHANNELS, N_TIMES = 2, 3, 30

N_TIMES_ATOM, N_ATOMS = 6, 4


@pytest.fixture
def X():
    rng = check_random_state(42)
    return rng.randn(N_TRIALS, N_CHANNELS, N_TIMES)


@pytest.fixture
def D_hat(X):
    return init_dictionary(X, N_ATOMS, N_TIMES_ATOM)


@pytest.fixture
def loss_params():
    return dict(gamma=1, sakoe_chiba_band=10, ordar=10)


@pytest.mark.parametrize('solver_z', ['l-bfgs', 'lgcd'])
@pytest.mark.parametrize('algorithm', ['batch', 'greedy', 'online',
                                       'stochastic'])
@pytest.mark.parametrize('reg', [None, 0.1])
@pytest.mark.parametrize('loss', ['l2', 'dwt'])
@pytest.mark.parametrize('uv_constraint', ['joint', 'separate'])
@pytest.mark.parametrize('feasible_evaluation', [True, False])
def test_get_encoder_for(solver_z, X, D_hat, algorithm, reg,
                         loss, loss_params, uv_constraint,
                         feasible_evaluation):
    """Test for valid values."""

    with get_z_encoder_for(solver=solver_z,
                           z_kwargs=None,
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           atom_support=N_TIMES_ATOM,
                           algorithm=algorithm,
                           reg=reg,
                           loss=loss,
                           loss_params=loss_params,
                           uv_constraint=uv_constraint,
                           feasible_evaluation=feasible_evaluation,
                           n_jobs=2,
                           use_sparse_z=False) as z_encoder:

        assert z_encoder is not None
        assert z_encoder.z_kwargs == dict()


@pytest.mark.parametrize('solver_z', [None, 'other'])
def test_get_encoder_for_error_solver_z(X, D_hat, loss_params, solver_z):
    """Tests for invalid values of `solver_z`."""

    with pytest.raises(ValueError) as error:
        get_z_encoder_for(solver=solver_z,
                          z_kwargs=None,
                          X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          algorithm='batch',
                          reg=None,
                          loss='l2',
                          loss_params=loss_params,
                          uv_constraint='joint',
                          feasible_evaluation=True,
                          n_jobs=2,
                          use_sparse_z=False)
        assert error.value.args[0] == f'unrecognized solver type: {solver_z}.'


def test_get_encoder_for_error_X(X, D_hat, loss_params):
    """Tests for invalid value of `X`."""
    with pytest.raises(ValueError) as error:
        get_z_encoder_for(solver='lgcd',
                          z_kwargs=None,
                          X=None,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          algorithm='batch',
                          reg=None,
                          loss='l2',
                          loss_params=loss_params,
                          uv_constraint='joint',
                          feasible_evaluation=True,
                          n_jobs=2,
                          use_sparse_z=False)
        assert error.value.args[
            0] == 'X should be a valid array of shape (n_trials, n_channels, n_times)'


def test_get_z_hat(X, D_hat, loss_params):
    """Test for valid values."""

    with get_z_encoder_for(solver='lgcd',
                           z_kwargs=None,
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           atom_support=N_TIMES_ATOM,
                           algorithm='batch',
                           reg=None,
                           loss='l2',
                           loss_params=loss_params,
                           uv_constraint='joint',
                           feasible_evaluation=True,
                           n_jobs=2,
                           use_sparse_z=False) as z_encoder:

        assert z_encoder is not None
        assert not z_encoder.get_z_hat().any()

        z_encoder.compute_z()
        assert z_encoder.get_z_hat().any()

    with get_z_encoder_for(solver='lgcd',
                           z_kwargs=None,
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           atom_support=N_TIMES_ATOM,
                           algorithm='batch',
                           reg=None,
                           loss='l2',
                           loss_params=loss_params,
                           uv_constraint='joint',
                           feasible_evaluation=True,
                           n_jobs=2,
                           use_sparse_z=True) as z_encoder:

        assert z_encoder is not None

        for matrix in z_encoder.get_z_hat():
            assert not matrix.count_nonzero()

        z_encoder.compute_z()
        for matrix in z_encoder.get_z_hat():
            assert matrix.count_nonzero()


def test_get_cost(X, D_hat, loss_params):

    with get_z_encoder_for(solver='lgcd',
                           z_kwargs=None,
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           atom_support=N_TIMES_ATOM,
                           algorithm='batch',
                           reg=None,
                           loss='l2',
                           loss_params=loss_params,
                           uv_constraint='joint',
                           feasible_evaluation=True,
                           n_jobs=2,
                           use_sparse_z=False) as z_encoder:
        assert not z_encoder.get_z_hat().any()


def test_compute_z(X, D_hat, loss_params):
    with get_z_encoder_for(solver='lgcd',
                           z_kwargs=None,
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           atom_support=N_TIMES_ATOM,
                           algorithm='batch',
                           reg=None,
                           loss='l2',
                           loss_params=loss_params,
                           uv_constraint='joint',
                           feasible_evaluation=True,
                           n_jobs=2,
                           use_sparse_z=False) as z_encoder:
        z_encoder.compute_z()
        assert z_encoder.get_z_hat().any()


def test_add_one_atom(X, D_hat, loss_params):

    with get_z_encoder_for(solver='lgcd',
                           z_kwargs=None,
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           atom_support=N_TIMES_ATOM,
                           algorithm='batch',
                           reg=None,
                           loss='l2',
                           loss_params=loss_params,
                           uv_constraint='joint',
                           feasible_evaluation=True,
                           n_jobs=2,
                           use_sparse_z=False) as z_encoder:
        new_atom = np.random.rand(N_CHANNELS + N_TIMES_ATOM)
        z_encoder.add_one_atom(new_atom)
        n_atoms_plus_one = z_encoder.D_hat.shape[0]
        assert n_atoms_plus_one == N_ATOMS + 1
