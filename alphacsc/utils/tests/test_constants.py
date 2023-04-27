import numpy as np


from alphacsc.utils.dictionary import get_D
from alphacsc.utils.validation import check_random_state
from alphacsc.utils.compute_constants import compute_DtD, compute_ztz
from alphacsc.utils.convolution import tensordot_convolve, construct_X_multi


def test_DtD():
    n_atoms = 10
    n_channels = 5
    n_times_atom = 50
    random_state = 42

    rng = check_random_state(random_state)

    uv = rng.randn(n_atoms, n_channels + n_times_atom)
    D = get_D(uv, n_channels)

    assert np.allclose(compute_DtD(uv, n_channels=n_channels),
                       compute_DtD(D))


def test_ztz():
    n_atoms = 7
    n_trials = 3
    n_channels = 5
    n_times_valid = 500
    n_times_atom = 10
    random_state = None

    rng = check_random_state(random_state)

    z = rng.randn(n_trials, n_atoms, n_times_valid)
    D = rng.randn(n_atoms, n_channels, n_times_atom)

    ztz = compute_ztz(z, n_times_atom)
    grad = tensordot_convolve(ztz, D)
    cost = np.dot(D.ravel(), grad.ravel())

    X_hat = construct_X_multi(z, D)

    assert np.isclose(cost, np.dot(X_hat.ravel(), X_hat.ravel()))
