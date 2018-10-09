import numpy as np

from alphacsc.learn_d_z_mcem import learn_d_z_weighted


def test_learn_d_z_weighted():
    """Test the output shapes."""
    n_atoms, n_times_atom = 3, 10
    n_trials, n_times = 2, 100
    rng = np.random.RandomState(0)
    X = rng.randn(n_trials, n_times)

    d_hat, z_hat, tau = learn_d_z_weighted(X, n_atoms, n_times_atom,
                                           random_state=0)
    assert d_hat.shape == (n_atoms, n_times_atom)
    assert z_hat.shape == (n_atoms, n_trials, n_times - n_times_atom + 1)
    assert tau.shape == (n_trials, n_times)
