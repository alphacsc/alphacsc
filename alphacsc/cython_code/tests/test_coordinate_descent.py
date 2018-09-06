import pytest
import numpy as np
from scipy import sparse
from numpy.testing import assert_allclose

from alphacsc.utils import check_random_state
from alphacsc.cython_code import _CYTHON_AVAILABLE
if not _CYTHON_AVAILABLE:
    pytest.skip("cython is not installed.", allow_module_level=True)
else:
    from alphacsc.cython_code.coordinate_descent import update_dz_opt


def test_dz_opt_sparse_update():
    n_trials = 10
    n_atoms = 10
    n_times_valid = 100

    t0, t1 = 20, 50
    density = 0.05

    rng = check_random_state(42)

    beta = rng.randn(n_atoms, n_times_valid)
    norm_D = rng.randn(n_atoms)
    reg = 1

    for _ in range(n_trials):
        random_state = rng.randint(65565)
        z = sparse.random(n_atoms, n_times_valid, density=density,
                          random_state=random_state, format='lil')
        # Check that we correctly handle the case where there is no tk > t1
        # in the sparse matrix
        z[-2, t1:] = 0
        # Check that we correctly handle the case where there is no nnz value
        # in the segment for z
        z[-1, t0:t1] = 0

        tmp = np.maximum(-beta - reg, 0) / norm_D[:, None]
        dz_opt = rng.randn(n_atoms, n_times_valid)
        dz_opt_expected = dz_opt.copy()

        dz_opt_expected[:, t0:t1] = tmp[:, t0:t1] - z[:, t0:t1]

        dz_opt = update_dz_opt(z, beta, dz_opt, norm_D, reg, t0, t1)

        assert_allclose(dz_opt[:, t0:t1], dz_opt_expected[:, t0:t1])
