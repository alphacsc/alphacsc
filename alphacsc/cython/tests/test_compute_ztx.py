import numpy as np
from scipy import sparse
from numpy.testing import assert_allclose

from alphacsc.cython import _fast_compute_ztx
from alphacsc.utils.compute_constants import compute_ZtX


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

    Z = sparse.random(n_atoms, n_trials * n_times_valid, density=density,
                      random_state=0)
    Z = Z.toarray().reshape(n_atoms, n_trials, n_times_valid)
    Z_lil = [sparse.lil_matrix(zi) for zi in np.swapaxes(Z, 0, 1)]

    ztx_0 = _fast_compute_ztx(Z_lil, X)
    ztx_1 = compute_ZtX(Z, X)
    assert_allclose(ztx_0, ztx_1, atol=1e-16)
