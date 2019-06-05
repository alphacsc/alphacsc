import pytest
import numpy as np
from functools import partial
from scipy.optimize import approx_fprime


from alphacsc.utils import get_D
from alphacsc.utils import construct_X_multi
from alphacsc.utils.whitening import whitening
from alphacsc.loss_and_gradient import gradient_d
from alphacsc.loss_and_gradient import gradient_zi
from alphacsc.loss_and_gradient import compute_X_and_objective_multi


def _gradient_zi(X, z, D, loss, loss_params, flatten=False):
    return gradient_zi(X[0], z[0], D, loss=loss, flatten=flatten,
                       loss_params=loss_params)


def _construct_X(X, z, D, loss, loss_params):
    return construct_X_multi(z, D, n_channels=X.shape[1])


def _objective(X, z, D, loss, loss_params):
    return compute_X_and_objective_multi(X, z, D, feasible_evaluation=False,
                                         loss=loss,
                                         loss_params=loss_params)


def _gradient_d(X, z, D, loss, loss_params, flatten=False):
    return gradient_d(D, X, z, loss=loss, flatten=flatten,
                      loss_params=loss_params)


def gradient_checker(func, grad, shape, args=(), kwargs={}, n_checks=10,
                     rtol=1e-5, grad_name='gradient', debug=False,
                     random_seed=None):
    """Check that the gradient correctly approximate the finite difference
    """

    rng = np.random.RandomState(random_seed)

    msg = ("Computed {} did not match gradient computed with finite "
           "difference. Relative error is {{}}".format(grad_name))

    func = partial(func, **kwargs)

    def test_grad(z0):
        grad_approx = approx_fprime(z0, func, 1e-8, *args)
        grad_compute = grad(z0, *args, **kwargs)
        error = np.sum((grad_approx - grad_compute) ** 2)
        error /= np.sum(grad_approx ** 2)
        error = np.sqrt(error)
        try:
            assert error < rtol, msg.format(error)
        except AssertionError:
            if debug:
                import matplotlib.pyplot as plt
                plt.plot(grad_approx)
                plt.plot(grad_compute)
                plt.show()
            raise

    z0 = np.zeros(shape)
    test_grad(z0)

    for _ in range(n_checks):
        z0 = rng.randn(shape)
        test_grad(z0)


@pytest.mark.parametrize('loss', ['l2', 'dtw', 'whitening'])
@pytest.mark.parametrize('func', [
    _construct_X, _gradient_zi, _objective, _gradient_d])
def test_consistency(loss, func):
    """Check that the result are the same for the full rank D and rank 1 uv.
    """
    n_trials, n_channels, n_times = 5, 3, 30
    n_atoms, n_times_atom = 4, 7

    loss_params = dict(gamma=.01)

    n_times_valid = n_times - n_times_atom + 1

    X = np.random.randn(n_trials, n_channels, n_times)
    z = np.random.randn(n_trials, n_atoms, n_times_valid)

    uv = np.random.randn(n_atoms, n_channels + n_times_atom)
    D = get_D(uv, n_channels)

    if loss == "whitening":
        loss_params['ar_model'], X = whitening(X)

    val_D = func(X, z, D, loss, loss_params=loss_params)
    val_uv = func(X, z, uv, loss, loss_params=loss_params)
    assert np.allclose(val_D, val_uv)


@pytest.mark.parametrize('loss', ['l2', 'dtw', 'whitening'])
def test_gradients(loss):
    """Check that the gradients have the correct shape.
    """
    n_trials, n_channels, n_times = 5, 3, 100
    n_atoms, n_times_atom = 10, 15

    n_checks = 5
    if loss == "dtw":
        n_checks = 1

    loss_params = dict(gamma=.01)

    n_times_valid = n_times - n_times_atom + 1

    X = np.random.randn(n_trials, n_channels, n_times)
    z = np.random.randn(n_trials, n_atoms, n_times_valid)

    uv = np.random.randn(n_atoms, n_channels + n_times_atom)
    D = get_D(uv, n_channels)
    if loss == "whitening":
        loss_params['ar_model'], X = whitening(X)

    # Test gradient D
    assert D.shape == _gradient_d(X, z, D, loss, loss_params=loss_params).shape

    def pobj(ds):
        return _objective(X, z, ds.reshape(n_atoms, n_channels, -1), loss,
                          loss_params=loss_params)

    def grad(ds):
        return _gradient_d(X, z, ds, loss=loss, flatten=True,
                           loss_params=loss_params)

    gradient_checker(pobj, grad, np.prod(D.shape), n_checks=n_checks,
                     grad_name="gradient D for loss '{}'".format(loss),
                     rtol=1e-4)

    # Test gradient z
    assert z[0].shape == _gradient_zi(
        X, z, D, loss, loss_params=loss_params).shape

    def pobj(zs):
        return _objective(X[:1], zs.reshape(1, n_atoms, -1), D, loss,
                          loss_params=loss_params)

    def grad(zs):
        return gradient_zi(X[0], zs, D, loss=loss, flatten=True,
                           loss_params=loss_params)

    gradient_checker(pobj, grad, n_atoms * n_times_valid, n_checks=n_checks,
                     debug=True, grad_name="gradient z for loss '{}'"
                     .format(loss), rtol=1e-4)
