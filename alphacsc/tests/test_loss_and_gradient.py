import pytest
import numpy as np


from alphacsc.utils import construct_X_multi
from alphacsc.utils.whitening import whitening
from alphacsc.loss_and_gradient import compute_X_and_objective_multi
from alphacsc.loss_and_gradient import gradient_d
from alphacsc.loss_and_gradient import gradient_zi


from alphacsc.utils import get_D


def _gradient_zi(X, Z, D, loss, loss_params):
    return gradient_zi(X[0], Z[:, 0], D, loss=loss,
                       loss_params=loss_params)


def _construct_X(X, Z, D, loss, loss_params):
    return construct_X_multi(Z, D, n_channels=X.shape[1])


def _objective(X, Z, D, loss, loss_params):
    return compute_X_and_objective_multi(X, Z, D, feasible_evaluation=False,
                                         loss=loss,
                                         loss_params=loss_params)


def _gradient_d(X, Z, D, loss, loss_params):
    return gradient_d(D, X, Z, loss=loss, loss_params=loss_params)


@pytest.mark.parametrize('loss', ['l2', 'dtw', 'whitening'])
@pytest.mark.parametrize('func', [
    _construct_X, _gradient_zi, _objective, _gradient_d])
def test_consistency(loss, func):
    """Check that the result are the same for the full rank D and rank 1 uv.
    """
    n_trials, n_channels, n_times = 5, 3, 100
    n_atoms, n_times_atom = 10, 15

    loss_params = dict(gamma=.01)

    n_times_valid = n_times - n_times_atom + 1

    X = np.random.randn(n_trials, n_channels, n_times)
    Z = np.random.randn(n_atoms, n_trials, n_times_valid)

    uv = np.random.randn(n_atoms, n_channels + n_times_atom)
    D = get_D(uv, n_channels)

    loss_params['ar_model'], X = whitening(X)

    val_D = func(X, Z, D, loss, loss_params=loss_params)
    val_uv = func(X, Z, uv, loss, loss_params=loss_params)
    assert np.allclose(val_D, val_uv)
