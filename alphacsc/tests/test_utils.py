from scipy import sparse
from numpy.testing import assert_allclose

from alphacsc.utils import _sparse_convolve, _dense_convolve, _choose_convolve
from alphacsc.utils import check_random_state


def test_sparse_convolve():
    rng = check_random_state(42)
    n_times = 128
    n_times_atoms = 21
    n_atoms = 3
    n_times_valid = n_times - n_times_atoms + 1
    density = 0.1
    zi = sparse.random(n_atoms, n_times_valid, density, random_state=rng)
    ds = rng.randn(n_atoms, n_times_atoms)
    zi = zi.toarray().reshape(n_atoms, n_times_valid)

    zd_0 = _dense_convolve(zi, ds)
    zd_1 = _sparse_convolve(zi, ds)
    zd_2 = _choose_convolve(zi, ds)
    assert_allclose(zd_0, zd_1, atol=1e-16)
    assert_allclose(zd_0, zd_2, atol=1e-16)
