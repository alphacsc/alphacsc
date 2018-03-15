import numpy as np

from .utils import check_random_state
from .update_d_multi import prox_uv


def init_uv(X, n_atoms, n_times_atom, uv_init=None, uv_constraint='separate',
            random_state=None):
    """Return an initial dictionary for the signals X

    Parameter
    ---------
    X: array, shape (n_trials, n_channels, n_times)
        The data on which to perform CSC.
    n_atoms : int
        The number of atoms to learn.
    n_times_atom : int
        The support of the atom.
    uv_init : array, shape (n_atoms, n_channels + n_times_atoms)
        The initial atoms.
    uv_constraint : str in {'joint', 'separate', 'box'}
        The kind of norm constraint on the atoms:
        If 'joint', the constraint is norm_2([u, v]) <= 1
        If 'separate', the constraint is norm_2(u) <= 1 and norm_2(v) <= 1
        If 'box', the constraint is norm_inf([u, v]) <= 1
    random_state : int | None
        The random state.

    Return
    ------
    uv: array shape (n_atoms, n_channels + n_times_atom)
        The initial atoms to learn from the data.
    """
    n_trials, n_channels, n_times = X.shape
    rng = check_random_state(random_state)

    if isinstance(uv_init, np.ndarray):
        uv_hat = uv_init.copy()
        assert uv_hat.shape == (n_atoms, n_channels + n_times_atom)

    elif uv_init is None or uv_init == "random":
        uv_hat = rng.randn(n_atoms, n_channels + n_times_atom)

    elif uv_init == 'chunk':
        u_hat = rng.randn(n_atoms, n_channels)
        v_hat = np.zeros((n_atoms, n_times_atom))
        for i_atom in range(n_atoms):
            i_trial = rng.randint(n_trials)
            i_channel = rng.randint(n_channels)
            t0 = rng.randint(n_times - n_times_atom)
            v_hat[i_atom] = X[i_trial, i_channel, t0:t0 + n_times_atom]
        uv_hat = np.c_[u_hat, v_hat]

    elif uv_init == "kmeans":
        raise NotImplementedError("Not yet")

    else:
        raise NotImplementedError('It is not possible to initialize uv with'
                                  ' parameter {}.'.format(uv_init))

    uv_hat = prox_uv(uv_hat, uv_constraint=uv_constraint, n_chan=n_channels)
    return uv_hat
