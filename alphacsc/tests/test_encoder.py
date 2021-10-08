import numpy as np

import pytest

from alphacsc._encoder import get_z_encoder_for
from alphacsc.init_dict import init_dictionary
from alphacsc.loss_and_gradient import compute_objective
from alphacsc.utils import check_random_state, construct_X_multi
from alphacsc.utils.compute_constants import compute_ztz, compute_ztX

N_CHANNELS, N_TIMES = 3, 30
N_TIMES_ATOM, N_ATOMS = 6, 4


@pytest.fixture
def rng():
    return check_random_state(42)


@pytest.fixture
def X(rng, n_trials):
    return rng.randn(n_trials, N_CHANNELS, N_TIMES)


@pytest.fixture
def D_hat(X, rank1):
    return init_dictionary(X,
                           N_ATOMS,
                           N_TIMES_ATOM,
                           random_state=0,
                           rank1=rank1
                           )


@pytest.fixture
def test_dicodile(solver_z):
    if solver_z == 'dicodile':
        return pytest.importorskip('dicodile')


@pytest.mark.parametrize('solver_z', ['l-bfgs', 'lgcd'])
@pytest.mark.parametrize('algorithm', ['batch', 'greedy', 'online',
                                       'stochastic'])
@pytest.mark.parametrize('loss', ['l2', 'dtw', 'whitening'])
@pytest.mark.parametrize('uv_constraint', ['joint', 'separate'])
@pytest.mark.parametrize('feasible_evaluation', [True, False])
@pytest.mark.parametrize('n_trials', [1, 2, 5])
@pytest.mark.parametrize('rank1', [True, False])
def test_get_encoder_for_alphacsc(X, solver_z, D_hat, algorithm, loss,
                                  uv_constraint, feasible_evaluation):
    """Test for valid values for alphacsc backend."""

    with get_z_encoder_for(solver=solver_z,
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           atom_support=N_TIMES_ATOM,
                           algorithm=algorithm,
                           loss=loss,
                           uv_constraint=uv_constraint,
                           feasible_evaluation=feasible_evaluation,
                           n_jobs=2) as z_encoder:

        assert z_encoder is not None


@pytest.mark.parametrize('solver_z, n_trials, rank1', [('dicodile', 1, False)])
def test_get_encoder_for_dicodile(X, D_hat, solver_z, test_dicodile):
    """Test for valid values for dicodile backend."""

    with get_z_encoder_for(solver=solver_z,
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           atom_support=N_TIMES_ATOM,
                           n_jobs=2) as z_encoder:

        assert z_encoder is not None


@pytest.mark.parametrize('solver_z, n_trials, rank1', [('dicodile', 2, False)])
def test_get_encoder_for_dicodile_error_n_trials(solver_z, X, D_hat,
                                                 test_dicodile):
    """Test for invalid n_trials value for dicodile backend."""

    with pytest.raises(AssertionError,
                       match=f"X should be a valid array of shape*"):
        get_z_encoder_for(solver=solver_z,
                          X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          n_jobs=2)


@pytest.mark.parametrize('solver_z, n_trials, rank1', [('dicodile', 1, True)])
def test_get_encoder_for_dicodile_error_rank1(X, D_hat, test_dicodile):
    """Test for invalid rank1 value for dicodile backend."""

    with pytest.raises(ValueError,
                       match=f"in1 and in2 should have the same dimensionality"):  # noqa
        get_z_encoder_for(solver='dicodile',
                          X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          n_jobs=2)


@pytest.mark.parametrize('n_trials', [2])
@pytest.mark.parametrize('rank1', [True])
@pytest.mark.parametrize('solver_z', [None, 'other'])
def test_get_encoder_for_error_solver_z(X, D_hat,  solver_z):
    """Tests for invalid values of `solver_z`."""

    with pytest.raises(ValueError,
                       match=f"unrecognized solver type: {solver_z}."):
        get_z_encoder_for(solver=solver_z,
                          X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          n_jobs=2)


@pytest.mark.parametrize('n_trials', [2])
@pytest.mark.parametrize('rank1', [True])
def test_get_encoder_for_error_z_kwargs(X, D_hat):
    """Tests for invalid value of `z_kwargs`."""

    with pytest.raises(AssertionError, match=".*z_kwargs should.*"):
        get_z_encoder_for(z_kwargs=None,
                          X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          n_jobs=2)


@pytest.mark.parametrize('n_trials', [2])
@pytest.mark.parametrize('rank1', [True])
@pytest.mark.parametrize('X_error', [None, np.zeros([2, N_CHANNELS])])
def test_get_encoder_for_error_X(X_error, D_hat):
    """Tests for invalid values of `X`."""

    with pytest.raises(AssertionError,
                       match="X should be a valid array of shape.*"):
        get_z_encoder_for(X=X_error,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          n_jobs=2)


@pytest.mark.parametrize('n_trials', [2])
@pytest.mark.parametrize('D_init', [None, np.zeros(2)])
def test_get_encoder_for_error_D_hat(X, D_init):
    """Tests for invalid values of `D_hat`."""

    with pytest.raises(AssertionError,
                       match="D_hat should be a valid array of shape.*"):
        get_z_encoder_for(X=X,
                          D_hat=D_init,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          n_jobs=2)


@pytest.mark.parametrize('n_trials', [2])
@pytest.mark.parametrize('rank1', [True])
@pytest.mark.parametrize('algorithm', [None, 'other'])
def test_get_encoder_for_error_algorithm(X, D_hat,  algorithm):
    """Tests for invalid values of `algorithm`."""

    with pytest.raises(AssertionError,
                       match=f"unrecognized algorithm type: {algorithm}"):
        get_z_encoder_for(X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          algorithm=algorithm,
                          n_jobs=2)


@pytest.mark.parametrize('n_trials', [2])
@pytest.mark.parametrize('rank1', [True])
def test_get_encoder_for_error_reg(X, D_hat):
    """Tests for invalid value of `reg`."""

    with pytest.raises(AssertionError,
                       match="reg value cannot be None."):
        get_z_encoder_for(X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          reg=None,
                          n_jobs=2)


@pytest.mark.parametrize('n_trials', [2])
@pytest.mark.parametrize('rank1', [True])
@pytest.mark.parametrize('loss', [None, 'other'])
def test_get_encoder_for_error_loss(X, D_hat,  loss):
    """Tests for invalid values of `loss`."""

    with pytest.raises(AssertionError,
                       match=f"unrecognized loss type: {loss}."):
        get_z_encoder_for(X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          loss=loss,
                          n_jobs=2)


@pytest.mark.parametrize('n_trials', [2])
@pytest.mark.parametrize('rank1', [True])
def test_get_encoder_for_error_loss_params(X, D_hat):
    """Tests for invalid value of `loss_params`."""

    with pytest.raises(AssertionError,
                       match="loss_params should be a valid dict or None."):
        get_z_encoder_for(X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          loss_params=42,
                          n_jobs=2)


@pytest.mark.parametrize('n_trials', [2])
@pytest.mark.parametrize('rank1', [True])
@pytest.mark.parametrize('uv_constraint', [None, 'other'])
def test_get_encoder_for_error_uv_constraint(X, D_hat,
                                             uv_constraint):
    """Tests for invalid values of `uv_constraint`."""

    with pytest.raises(AssertionError,
                       match="unrecognized uv_constraint type.*"):
        get_z_encoder_for(X=X,
                          D_hat=D_hat,
                          n_atoms=N_ATOMS,
                          atom_support=N_TIMES_ATOM,
                          uv_constraint=uv_constraint,
                          n_jobs=2)


@pytest.mark.parametrize('solver_z, n_trials, rank1',
                         [('l-bfgs', 3, True),
                          #                          ('dicodile', 1, False)
                          ])
def test_get_z_hat(solver_z, X, D_hat, test_dicodile):
    """Test for valid values."""

    with get_z_encoder_for(solver=solver_z,
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


@pytest.mark.parametrize('n_trials', [1, 3])
@pytest.mark.parametrize('rank1', [True, False])
def test_get_z_hat_use_sparse_z(X, D_hat):
    """Test for valid values when use_sparse_z=True."""
    with get_z_encoder_for(solver='lgcd',
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


@pytest.mark.parametrize('solver_z, n_trials, rank1', [('l-bfgs', 3, True),
                                                       #                                                       ('dicodile', 1, False)
                                                       ])
def test_get_cost(solver_z, X, D_hat, test_dicodile):
    """Test for valid values."""

    with get_z_encoder_for(solver=solver_z,
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           atom_support=N_TIMES_ATOM,
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


@pytest.mark.parametrize('solver_z, n_trials, rank1', [('lgcd', 2, True),
                                                       ('l-bfgs', 5, False),
                                                       ('dicodile', 1, False)])
def test_compute_z(solver_z, X, D_hat, test_dicodile):
    """Test for valid values."""

    with get_z_encoder_for(solver=solver_z,
                           X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           atom_support=N_TIMES_ATOM,
                           n_jobs=2) as z_encoder:
        z_encoder.compute_z()
        assert z_encoder.get_z_hat().any()


@pytest.mark.parametrize('n_trials', [2])
@pytest.mark.parametrize('rank1', [True])
def test_compute_z_partial(X, D_hat, n_trials, rng):
    """Test for valid values."""

    with get_z_encoder_for(X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           atom_support=N_TIMES_ATOM,
                           n_jobs=2) as z_encoder:

        i0 = rng.choice(n_trials, 1, replace=False)
        z_encoder.compute_z_partial(i0)
        assert z_encoder.get_z_hat().any()


@pytest.mark.parametrize('solver_z, n_trials, rank1', [('lgcd', 2, True),
                                                       ('l-bfgs', 5, False),
                                                       ('dicodile', 1, False)])
def test_get_sufficient_statistics(solver_z, X, D_hat, test_dicodile):
    """Test for valid values."""

    z_encoder = get_z_encoder_for(solver=solver_z,
                                  X=X,
                                  D_hat=D_hat,
                                  n_atoms=N_ATOMS,
                                  atom_support=N_TIMES_ATOM,
                                  n_jobs=2)

    z_encoder.compute_z()
    z_hat = z_encoder.get_z_hat()

    ztz, ztX = z_encoder.get_sufficient_statistics()
    assert ztz is not None and np.allclose(ztz, compute_ztz(z_hat,
                                                            N_TIMES_ATOM))

    assert ztX is not None and np.allclose(ztX, compute_ztX(z_hat, X))


@pytest.mark.parametrize('solver_z, n_trials, rank1', [('lgcd', 2, True),
                                                       ('l-bfgs', 5, False),
                                                       #                                                       ('dicodile', 1, False)
                                                       ])
def test_get_sufficient_statistics_error(solver_z, X, D_hat, test_dicodile):
    """Test for invalid call to function."""

    z_encoder = get_z_encoder_for(solver=solver_z,
                                  X=X,
                                  D_hat=D_hat,
                                  n_atoms=N_ATOMS,
                                  atom_support=N_TIMES_ATOM,
                                  n_jobs=2)

    # test before calling compute_z
    with pytest.raises(AssertionError,
                       match="compute_z should be called.*"):
        z_encoder.get_sufficient_statistics()


@pytest.mark.parametrize('n_trials', [2])
@pytest.mark.parametrize('rank1', [True])
def test_get_sufficient_statistics_partial(X, D_hat, n_trials, rng):
    """Test for valid values."""

    z_encoder = get_z_encoder_for(X=X,
                                  D_hat=D_hat,
                                  n_atoms=N_ATOMS,
                                  atom_support=N_TIMES_ATOM,
                                  n_jobs=2)

    i0 = rng.choice(n_trials, 1, replace=False)
    z_encoder.compute_z_partial(i0)

    ztz_i0, ztX_i0 = z_encoder.get_sufficient_statistics_partial()
    assert ztz_i0 is not None and ztX_i0 is not None


@pytest.mark.parametrize('n_trials', [2])
@pytest.mark.parametrize('rank1', [True])
def test_get_sufficient_statistics_partial_error(X, D_hat):
    """Test for invalid call to function."""

    z_encoder = get_z_encoder_for(X=X,
                                  D_hat=D_hat,
                                  n_atoms=N_ATOMS,
                                  atom_support=N_TIMES_ATOM,
                                  n_jobs=2)

    # test before calling compute_z_partial
    with pytest.raises(AssertionError,
                       match="compute_z_partial should be called.*"):
        z_encoder.get_sufficient_statistics_partial()


@pytest.mark.parametrize('n_trials', [2])
@pytest.mark.parametrize('rank1', [True])
def test_add_one_atom(X, D_hat):
    """Test for valid values."""

    with get_z_encoder_for(X=X,
                           D_hat=D_hat,
                           n_atoms=N_ATOMS,
                           atom_support=N_TIMES_ATOM,
                           n_jobs=2) as z_encoder:
        new_atom = np.random.rand(N_CHANNELS + N_TIMES_ATOM)
        z_encoder.add_one_atom(new_atom)
        n_atoms_plus_one = z_encoder.D_hat.shape[0]
        assert n_atoms_plus_one == N_ATOMS + 1
