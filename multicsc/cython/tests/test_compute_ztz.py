import numpy as np
from scipy import sparse
from numpy.testing import assert_allclose

from multicsc.cython import _fast_compute_ztz
from multicsc.utils.compute_constants import compute_ZtZ
from multicsc.utils.lil import convert_to_list_of_lil


def test_sparse_convolve():
    n_times = 128
    n_times_atom = 21
    n_atoms = 3
    n_times_valid = n_times - n_times_atom + 1
    density = 0.1
    n_trials = 4
    Z = sparse.random(n_trials, n_atoms * n_times_valid, density=density,
                      random_state=0)
    Z = Z.toarray().reshape(n_trials, n_atoms, n_times_valid)
    Z_lil = convert_to_list_of_lil(Z)

    ztz_0 = _fast_compute_ztz(Z_lil, n_times_atom)
    ztz_1 = compute_ZtZ(Z, n_times_atom)
    assert_allclose(ztz_0, ztz_1, atol=1e-16)
