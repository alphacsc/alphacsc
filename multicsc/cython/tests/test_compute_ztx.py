import numpy as np
from scipy import sparse
from numpy.testing import assert_allclose

from multicsc.cython import _fast_compute_ztx
from multicsc.utils.compute_constants import compute_ZtX
from multicsc.utils.lil import convert_to_list_of_lil


def test_sparse_convolve():
    n_times = 128
    n_times_atom = 21
    n_channels = 2
    n_atoms = 3
    n_times_valid = n_times - n_times_atom + 1
    density = 0.1
    n_trials = 4
    rng = np.random.RandomState(0)
    X = rng.randn(n_trials, n_channels, n_times)

    Z = sparse.random(n_trials, n_atoms * n_times_valid, density=density,
                      random_state=0)
    Z = Z.toarray().reshape(n_trials, n_atoms, n_times_valid)
    Z_lil = convert_to_list_of_lil(Z)

    ztx_0 = _fast_compute_ztx(Z_lil, X)
    ztx_1 = compute_ZtX(Z, X)
    assert_allclose(ztx_0, ztx_1, atol=1e-16)
