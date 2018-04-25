from scipy import sparse
from numpy.testing import assert_allclose

from alphacsc.utils import check_random_state
from alphacsc.cython.coordinate_descent import update_dz_opt


def test_dz_opt_sparse_update():
    n_trials = 10
    n_atoms = 10
    n_times_valid = 100

    t0, t1 = 20, 50
    density = 0.05

    rng = check_random_state(42)

    for _ in range(n_trials):
        random_state = rng.randint(65565)
        Z = sparse.random(n_atoms, n_times_valid, density=density,
                          random_state=random_state, format='lil')
        # Check that we correctly handle the case where there is no tk > t1
        # in the sparse matrix
        Z[-2, t1:] = 0
        # Check that we correctly handle the case where there is no nnz value
        # in the segment for z
        Z[-1, t0:t1] = 0

        tmp = rng.randn(n_atoms, t1 - t0)
        dz_opt = rng.randn(n_atoms, n_times_valid)
        dz_opt_expected = dz_opt.copy()

        dz_opt_expected[:, t0:t1] = tmp - Z[:, t0:t1]

        dz_opt = update_dz_opt(Z, tmp, dz_opt, t0, t1)

        assert_allclose(dz_opt[:, t0:t1], dz_opt_expected[:, t0:t1])

