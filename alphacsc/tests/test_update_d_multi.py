import numpy as np
from scipy import optimize, signal

from alphacsc.loss_and_gradient import compute_objective
from alphacsc.loss_and_gradient import gradient_d, gradient_uv
from alphacsc.update_d_multi import _get_d_update_constants
from alphacsc.utils.convolution import construct_X_multi


DEBUG = True


def test_simple():
    T = 100
    L = 10
    S = T - L + 1
    x = np.random.random(T)
    z = np.random.random(S)
    d = np.random.random(L)

    def func(d0):
        xr = signal.convolve(z, d0)
        residual = x - xr
        return .5 * np.sum(residual * residual)

    def grad(d0):
        xr = signal.convolve(z, d0)
        residual = x - xr
        grad_d = - signal.convolve(residual, z[::-1], mode='valid')
        return grad_d

    error = optimize.check_grad(func, grad, d, epsilon=1e-8)
    assert error < 1e-4, "Gradient is false: {:.4e}".format(error)


def test_gradient_d():
    # Generate synchronous D
    n_times_atom, n_times = 10, 100
    n_channels = 5
    n_atoms = 2
    n_trials = 3

    rng = np.random.RandomState()
    X = rng.normal(size=(n_trials, n_channels, n_times))
    z = rng.normal(size=(n_trials, n_atoms, n_times - n_times_atom + 1))
    d = rng.normal(size=(n_atoms, n_channels, n_times_atom)).ravel()

    def func(d0):
        D0 = d0.reshape(n_atoms, n_channels, n_times_atom)
        X_hat = construct_X_multi(z, D=D0)
        return compute_objective(X, X_hat)

    def grad(d0):
        return gradient_d(D=d0, X=X, z=z, flatten=True)

    error = optimize.check_grad(func, grad, d, epsilon=2e-8)
    grad_d = grad(d)
    n_grad = np.sqrt(np.dot(grad_d, grad_d))
    try:
        assert error < 1e-5 * n_grad, "Gradient is false: {:.4e}".format(error)
    except AssertionError:
        if DEBUG:
            grad_approx = optimize.approx_fprime(d, func, 2e-8)

            import matplotlib.pyplot as plt
            plt.semilogy(abs(grad_approx - grad_d))
            plt.figure()
            plt.plot(grad_approx, label="approx")
            plt.plot(grad_d, '--', label="grad")
            plt.legend()
            plt.show()
        raise


def test_gradient_uv():
    # Generate synchronous D
    n_times_atom, n_times = 10, 100
    n_channels = 5
    n_atoms = 2
    n_trials = 3

    rng = np.random.RandomState()
    X = rng.normal(size=(n_trials, n_channels, n_times))
    z = rng.normal(size=(n_trials, n_atoms, n_times - n_times_atom + 1))
    uv = rng.normal(size=(n_atoms, n_channels + n_times_atom)).ravel()

    def func(uv0):
        uv0 = uv0.reshape(n_atoms, n_channels + n_times_atom)
        X_hat = construct_X_multi(z, D=uv0, n_channels=n_channels)
        return compute_objective(X, X_hat)

    def grad(uv0):
        return gradient_uv(uv=uv0, X=X, z=z, flatten=True)

    error = optimize.check_grad(func, grad, uv.ravel(), epsilon=2e-8)
    grad_uv = grad(uv)
    n_grad = np.sqrt(np.dot(grad_uv, grad_uv))
    try:
        assert error < 1e-5 * n_grad, "Gradient is false: {:.4e}".format(error)
    except AssertionError:

        if DEBUG:
            grad_approx = optimize.approx_fprime(uv, func, 2e-8)

            import matplotlib.pyplot as plt
            plt.semilogy(abs(grad_approx - grad_uv))
            plt.figure()
            plt.plot(grad_approx, label="approx")
            plt.plot(grad_uv, '--', label="grad")
            plt.legend()
            plt.show()
        raise

    constants = _get_d_update_constants(X, z)
    msg = "Wrong value for zt*X"
    assert np.allclose(
        gradient_uv(0 * uv, X=X, z=z, flatten=True),
        gradient_uv(0 * uv, constants=constants, flatten=True)), msg
    msg = "Wrong value for zt*z"
    assert np.allclose(
        gradient_uv(uv, X=X, z=z, flatten=True),
        gradient_uv(uv, constants=constants, flatten=True)), msg


def test_fast_cost():
    """Test that _shifted_objective_uv compute the right thing"""
    # Generate synchronous D
    n_times_atom, n_times = 10, 40
    n_channels = 3
    n_atoms = 2
    n_trials = 4

    rng = np.random.RandomState()
    X = rng.normal(size=(n_trials, n_channels, n_times))
    z = rng.normal(size=(n_trials, n_atoms, n_times - n_times_atom + 1))

    constants = _get_d_update_constants(X, z)

    def objective(uv):
        X_hat = construct_X_multi(z, D=uv, n_channels=n_channels)
        res = X - X_hat
        return .5 * np.sum(res * res)

    for _ in range(5):
        uv = rng.normal(size=(n_atoms, n_channels + n_times_atom))

        cost_fast = compute_objective(D=uv, constants=constants)
        cost_full = objective(uv)
        assert np.isclose(cost_full, cost_fast)


def test_constants_d():
    """Test that _shifted_objective_uv compute the right thing"""
    # Generate synchronous D
    n_times_atom, n_times = 10, 100
    n_channels = 5
    n_atoms = 2
    n_trials = 3

    rng = np.random.RandomState()
    X = rng.normal(size=(n_trials, n_channels, n_times))
    z = rng.normal(size=(n_trials, n_atoms, n_times - n_times_atom + 1))

    from alphacsc.update_d_multi import _get_d_update_constants
    constants = _get_d_update_constants(X, z)

    ztX = np.sum([[[np.convolve(zik[::-1], xip, mode='valid') for xip in xi]
                   for zik in zi] for zi, xi in zip(z, X)], axis=0)

    assert np.allclose(ztX, constants['ztX'])

    ztz = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    t0 = n_times_atom - 1
    axes = ([0, 2], [0, 2])

    for t in range(n_times_atom):
        if t == 0:
            ztz[:, :, t0] += np.tensordot(z, z, axes=axes)
        else:
            tmp = np.tensordot(z[:, :, :-t], z[:, :, t:], axes=axes)
            ztz[:, :, t0 + t] += tmp
            tmp = np.tensordot(z[:, :, t:], z[:, :, :-t], axes=axes)
            ztz[:, :, t0 - t] += tmp

    assert np.allclose(ztz, constants['ztz'])
