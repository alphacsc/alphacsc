import numpy as np


from alphacsc.utils.compute_constants import _compute_DtD
from alphacsc.utils import check_random_state, get_D


def test_DtD():
    n_atoms = 10
    n_channels = 5
    n_times_atom = 50
    random_state = 42

    rng = check_random_state(random_state)

    uv = rng.randn(n_atoms, n_channels + n_times_atom)
    D = get_D(uv, n_channels)

    assert np.allclose(_compute_DtD(uv, n_channels=n_channels),
                       _compute_DtD(D))
