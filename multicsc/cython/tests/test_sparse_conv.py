import numpy as np
from scipy import sparse
from numpy.testing import assert_allclose

from multicsc.utils.convolution import _dense_convolve_multi
from multicsc.utils.convolution import _dense_convolve_multi_uv
from multicsc.utils import check_random_state
from multicsc.cython import _fast_sparse_convolve_multi
from multicsc.cython import _fast_sparse_convolve_multi_uv


def test_sparse_convolve():
    rng = check_random_state(42)
    n_times = 128
    n_channels = 5
    n_times_atom = 21
    n_atoms = 3
    n_times_valid = n_times - n_times_atom + 1
    density = 0.1
    zi_lil = sparse.random(n_atoms, n_times_valid, density, format='lil',
                           random_state=rng)
    ds = rng.randn(n_atoms, n_channels, n_times_atom)
    zi = zi_lil.toarray().reshape(n_atoms, n_times_valid)

    zd_0 = _dense_convolve_multi(zi, ds)
    zd_1 = np.zeros_like(zd_0)
    zd_1 = _fast_sparse_convolve_multi(zi_lil, ds)
    assert_allclose(zd_0, zd_1, atol=1e-16)


def test_sparse_convolve_uv():
    rng = check_random_state(42)
    n_times = 128
    n_channels = 5
    n_times_atom = 21
    n_atoms = 3
    n_times_valid = n_times - n_times_atom + 1
    density = 0.1
    zi_lil = sparse.random(n_atoms, n_times_valid, density, format='lil',
                           random_state=rng)
    ds = rng.randn(n_atoms, n_channels + n_times_atom)
    zi = zi_lil.toarray().reshape(n_atoms, n_times_valid)

    zd_0 = _dense_convolve_multi_uv(zi, ds, n_channels)
    zd_1 = np.zeros_like(zd_0)
    zd_1 = _fast_sparse_convolve_multi_uv(zi_lil, ds, n_channels)
    assert_allclose(zd_0, zd_1, atol=1e-16)