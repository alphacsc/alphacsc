import pytest
import numpy as np
from scipy import sparse
from numpy.testing import assert_allclose

from alphacsc.utils.compute_constants import compute_ztX
from alphacsc.utils.lil import convert_to_list_of_lil

from alphacsc import cython_code
if not cython_code._CYTHON_AVAILABLE:
    pytest.skip("cython is not installed.", allow_module_level=True)


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

    z = sparse.random(n_trials, n_atoms * n_times_valid, density=density,
                      random_state=0)
    z = z.toarray().reshape(n_trials, n_atoms, n_times_valid)
    z_lil = convert_to_list_of_lil(z)

    ztX_0 = cython_code._fast_compute_ztX(z_lil, X)
    ztX_1 = compute_ztX(z, X)
    assert_allclose(ztX_0, ztX_1, atol=1e-16)
