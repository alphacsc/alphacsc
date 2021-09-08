import numpy as np

import pytest

from alphacsc._encoder import get_z_encoder_for
from alphacsc.init_dict import init_dictionary
from alphacsc.utils import check_random_state


def test_add_one_atom():
    n_trials, n_channels, n_times = 2, 3, 30
    n_times_atom, n_atoms = 6, 4

    loss_params = dict(gamma=1, sakoe_chiba_band=10, ordar=10)

    rng = check_random_state(42)
    X = rng.randn(n_trials, n_channels, n_times)

    D_hat = init_dictionary(X, n_atoms, n_times_atom)

    with get_z_encoder_for(solver='lgcd', z_kwargs=None, X=X,
                           D_hat=D_hat, n_atoms=n_atoms,
                           atom_support=n_times_atom, algorithm='batch',
                           reg=None, loss='l2',
                           loss_params=loss_params, uv_constraint='joint',
                           feasible_evaluation=True,
                           n_jobs=2, use_sparse_z=False) as z_encoder:
        new_atom = np.random.rand(n_channels + n_times_atom)
        z_encoder.add_one_atom(new_atom)
        n_atoms_plus_one = z_encoder.D_hat.shape[0]
        assert n_atoms_plus_one == n_atoms + 1
