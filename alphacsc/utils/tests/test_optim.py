import numpy as np

from alphacsc.utils.optim import fista, power_iteration


def test_ista():
    """Test that objective goes down in ISTA for a simple problem."""

    # || Ax - b ||_2^2
    n, p = 100, 10
    x = np.random.randn(p)
    x /= np.linalg.norm(x)
    A = np.random.randn(n, p)
    b = np.dot(A, x)

    def obj(x):
        res = A.dot(x) - b
        return 0.5 * np.dot(res.ravel(), res.ravel())

    def grad(x):
        return A.T.dot(A.dot(x) - b)

    def prox(x, step_size=0):
        return x / max(np.linalg.norm(x), 1.)

    x0 = np.random.rand(p)
    L = power_iteration(A.dot(A.T))
    step_size = 0.99 / L
    x_hat, pobj = fista(obj, grad, prox, step_size, x0, max_iter=600,
                        momentum=False, eps=None, debug=True, verbose=0)
    np.testing.assert_array_almost_equal(x, x_hat)
    assert np.all(np.diff(pobj) <= 0)


def test_power_iterations():
    """Test power iteration."""
    A = np.diag((1, 2, 3))
    mu, b = np.linalg.eig(A)
    mu_hat = power_iteration(A)
    assert np.isclose(mu_hat, mu.max())
