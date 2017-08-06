import time
import unittest
from functools import partial

import numpy as np
from numpy.testing import assert_allclose
from nose.tools import assert_equal, assert_true
from scipy import linalg

from alphacsc.update_z import update_z, power_iteration, gram_block_circulant
from alphacsc.update_d import update_d, update_d_block
from alphacsc.update_d import solve_unit_norm_dual, solve_unit_norm_primal
from alphacsc.learn_d_z import learn_d_z

from alphacsc.utils import construct_X, check_random_state
from alphacsc.simulate import simulate_data

_dummy = unittest.TestCase('__init__')
assert_raises = _dummy.assertRaises

n_trials = 10
reg = 0.1  # lambda
n_times_atom = 16  # q
n_times = 128  # T
n_atoms = 2


def test_learn_atoms():
    """Test learning of atoms."""
    X, ds, z = simulate_data(n_trials, n_times, n_times_atom, n_atoms)
    d_hat, _ = update_d(X, z, n_times_atom)

    assert_allclose(ds, d_hat)

    X_hat = construct_X(z, d_hat)
    assert_allclose(X, X_hat, rtol=1e-05, atol=1e-12)


def test_learn_codes():
    """Test learning of codes."""
    thresh = 0.25

    X, ds, z = simulate_data(n_trials, n_times, n_times_atom, n_atoms)

    for solver in ('l_bfgs', 'ista', 'fista'):
        z_hat = update_z(X, ds, reg, n_times_atom, solver=solver,
                         solver_kwargs=dict(factr=1e11, max_iter=50))

        X_hat = construct_X(z_hat, ds)
        assert_true(np.corrcoef(X.ravel(), X_hat.ravel())[1, 1] > 0.99)
        assert_true(np.max(X - X_hat) < 0.1)

        # Find position of non-zero entries
        idx = np.ravel_multi_index(z[0].nonzero(), z[0].shape)
        loc_x, loc_y = np.where(z_hat[0] > thresh)
        # shift position by half the length of atom
        idx_hat = np.ravel_multi_index((loc_x, loc_y), z_hat[0].shape)
        # make sure that the positions are a subset of the positions
        # in the original z
        mask = np.in1d(idx_hat, idx)
        assert_equal(np.sum(mask), len(mask))


def test_learn_codes_atoms():
    """Test that the objective function is decreasing."""
    random_state = 1
    n_iter = 10
    X, ds, z = simulate_data(n_trials, n_times, n_times_atom, n_atoms)
    func_d_0 = partial(update_d_block, projection='dual', n_iter=5)
    func_d_1 = partial(update_d_block, projection='primal', n_iter=5)
    for func_d in [func_d_0, func_d_1, update_d]:
        for solver_z in ('l_bfgs', 'ista', 'fista'):
            print(func_d, solver_z)
            pobj, times, d_hat, _ = learn_d_z(
                X, n_atoms, n_times_atom, func_d=func_d, solver_z=solver_z,
                reg=reg, n_iter=n_iter, verbose=0, random_state=random_state,
                solver_z_kwargs=dict(factr=1e7, max_iter=200))
            assert_true(np.all(np.diff(pobj) < 0))


def test_solve_unit_norm():
    """Test solving constraint for ||d||^2 <= 1."""
    rng = check_random_state(42)

    n, p = 10, 10
    x = np.zeros(p)
    x[3] = 3.
    A = rng.randn(n, p)
    b = np.dot(A, x)
    lhs = np.dot(A.T, A)
    rhs = np.dot(A.T, b)

    x_hat, lambd_hat = solve_unit_norm_dual(lhs, rhs,
                                            np.array([10.]), debug=True)
    # warm start
    x_hat2, _ = solve_unit_norm_dual(lhs, rhs, lambd0=lambd_hat)

    assert_true(linalg.norm(x_hat) - 1. < 1e-3)
    assert_true(linalg.norm(x_hat2) - 1. < 1e-3)

    x_hat = solve_unit_norm_primal(lhs, rhs, d_hat0=rng.randn(p))
    assert_true(linalg.norm(x_hat) - 1. < 1e-3)

    # back to dual, for more than one atom
    x[7] = 5
    x_hat, lambd_hat = solve_unit_norm_dual(lhs, rhs, np.array([5., 10.]))
    assert_true(linalg.norm(x_hat[:5]) - 1. < 1e-3)


def test_power_iterations():
    """Test power iteration."""
    A = np.diag((1, 2, 3))
    mu, b = np.linalg.eig(A)
    mu_hat = power_iteration(A)
    assert_allclose(mu_hat, mu.max())


def test_linear_operator():
    """Test linear operator."""
    n_times, n_atoms, n_times_atom = 128, 32, 32
    n_times_valid = n_times - n_times_atom + 1

    rng = check_random_state(42)
    ds = rng.randn(n_atoms, n_times_atom)
    some_sample_weights = np.abs(rng.randn(n_times))

    for sample_weights in [None, some_sample_weights]:
        gbc = partial(gram_block_circulant, ds=ds, n_times_valid=n_times_valid,
                      sample_weights=sample_weights)
        DTD_full = gbc(method='full')
        DTD_scipy = gbc(method='scipy')
        DTD_custom = gbc(method='custom')

        z = rng.rand(DTD_full.shape[1])
        assert_allclose(DTD_full.dot(z), DTD_scipy.dot(z))
        assert_allclose(DTD_full.dot(z), DTD_custom.dot(z))

        # test power iterations with linear operator
        mu, _ = linalg.eigh(DTD_full)
        t = []
        for DTD in [DTD_full, DTD_scipy, DTD_custom]:
            start = time.time()
            mu_hat = power_iteration(DTD)
            t.append(time.time() - start)
            assert_allclose(np.max(mu), mu_hat, rtol=1e-2)

        print(t)
    assert_true(t[1] < t[0])
    assert_true(t[2] < t[0])


def test_update_d():
    """Test vanilla d update."""
    rng = check_random_state(42)
    X, ds, z = simulate_data(n_trials, n_times, n_times_atom, n_atoms)
    ds_init = rng.randn(n_atoms, n_times_atom)

    # This number of iteration is 1 in the general case, but needs to be
    # increased to compare with update_d
    n_iter_d_block = 5

    # All solvers should give the same results
    d_hat_0, _ = update_d(X, z, n_times_atom, lambd0=None, ds_init=ds_init)
    d_hat_1, _ = update_d_block(X, z, n_times_atom, lambd0=None,
                                ds_init=ds_init, n_iter=n_iter_d_block)
    assert_allclose(d_hat_0, d_hat_1, rtol=1e-5)


def test_update_z_sample_weights():
    """Test z update with weights."""
    rng = check_random_state(42)
    X, ds, z = simulate_data(n_trials, n_times, n_times_atom, n_atoms)
    b_hat_0 = rng.randn(n_atoms * (n_times - n_times_atom + 1))

    # Having sample_weights all identical is equivalent to having
    # sample_weights=None and a scaled regularization
    factor = 1.6
    sample_weights = np.ones_like(X) * factor
    for solver in ('l_bfgs', 'ista', 'fista'):
        z_0 = update_z(X, ds, reg * factor, n_times_atom, solver=solver,
                       solver_kwargs=dict(factr=1e7, max_iter=50),
                       b_hat_0=b_hat_0.copy(), sample_weights=sample_weights)
        z_1 = update_z(X, ds, reg, n_times_atom, solver=solver,
                       solver_kwargs=dict(factr=1e7, max_iter=50),
                       b_hat_0=b_hat_0.copy(), sample_weights=None)
        assert_allclose(z_0, z_1, rtol=1e-4)

    # All solvers should give the same results
    sample_weights = np.abs(rng.randn(*X.shape))
    sample_weights /= sample_weights.mean()
    z_list = []
    for solver in ('l_bfgs', 'ista', 'fista'):
        z_hat = update_z(X, ds, reg, n_times_atom, solver=solver,
                         solver_kwargs=dict(factr=1e7, max_iter=2000),
                         b_hat_0=b_hat_0.copy(), sample_weights=sample_weights)
        z_list.append(z_hat)
    assert_allclose(z_list[0][z != 0], z_list[1][z != 0], rtol=1e-3)
    assert_allclose(z_list[0][z != 0], z_list[2][z != 0], rtol=1e-3)

    # And using no sample weights should give different results
    z_hat = update_z(X, ds, reg, n_times_atom, solver=solver,
                     solver_kwargs=dict(factr=1e7, max_iter=2000),
                     b_hat_0=b_hat_0.copy(), sample_weights=None)
    assert_raises(AssertionError, assert_allclose, z_list[0][z != 0],
                  z_hat[z != 0], 1e-3)


def test_update_d_sample_weights():
    """Test d update with weights."""
    rng = check_random_state(42)
    X, ds, z = simulate_data(n_trials, n_times, n_times_atom, n_atoms)
    ds_init = rng.randn(n_atoms, n_times_atom)
    # we need noise to have different results with different sample_weights
    X += 0.1 * rng.randn(*X.shape)

    # This number of iteration is 1 in the general case, but needs to be
    # increased to compare with update_d
    n_iter = 5
    func_d_0 = partial(update_d_block, projection='dual', n_iter=n_iter)
    func_d_1 = partial(update_d_block, projection='primal', n_iter=n_iter,
                       solver_kwargs=dict(factr=1e3))
    func_d_list = [func_d_0, func_d_1, update_d]

    # Having sample_weights all identical is equivalent to having
    # sample_weights=None
    factor = 1.6
    sample_weights = np.ones_like(X) * factor
    for func_d in func_d_list:
        d_hat_0, _ = func_d(X, z, n_times_atom, lambd0=None, ds_init=ds_init,
                            sample_weights=sample_weights)
        d_hat_1, _ = func_d(X, z, n_times_atom, lambd0=None, ds_init=ds_init,
                            sample_weights=None)
        assert_allclose(d_hat_0, d_hat_1, rtol=1e-5)

    # All solvers should give the same results
    sample_weights = np.abs(rng.randn(*X.shape))
    sample_weights /= sample_weights.mean()
    d_hat_list = []
    for func_d in func_d_list:
        d_hat, _ = func_d(X, z, n_times_atom, lambd0=None, ds_init=ds_init,
                          sample_weights=sample_weights)
        d_hat_list.append(d_hat)
    for d_hat in d_hat_list[1:]:
        assert_allclose(d_hat, d_hat_list[0], rtol=1e-5)

    # And using no sample weights should give different results
    for func_d in func_d_list:
        d_hat_2, _ = func_d(X, z, n_times_atom, lambd0=None, ds_init=ds_init,
                            sample_weights=None)
        assert_raises(AssertionError, assert_allclose, d_hat, d_hat_2, 1e-7)


def test_learn_codes_atoms_sample_weights():
    """Test weighted CSC."""
    rng = check_random_state(42)
    X, ds, z = simulate_data(n_trials, n_times, n_times_atom, n_atoms)
    ds_init = rng.randn(n_atoms, n_times_atom)
    X += 0.1 * rng.randn(*X.shape)
    n_iter = 3
    reg = 0.1

    func_d_0 = partial(update_d_block, projection='dual')
    func_d_1 = partial(update_d_block, projection='primal')
    func_d_list = [func_d_0, func_d_1, update_d]

    # sample_weights all equal to one is equivalent to sample_weights=None.
    sample_weights = np.ones_like(X)
    for func_d in func_d_list:
        for solver_z in ('l_bfgs', 'ista', 'fista'):
            pobj_0, _, _, _ = learn_d_z(
                X, n_atoms, n_times_atom, func_d=func_d, solver_z=solver_z,
                reg=reg, n_iter=n_iter, random_state=0, verbose=0,
                sample_weights=sample_weights, ds_init=ds_init)
            pobj_1, _, _, _ = learn_d_z(
                X, n_atoms, n_times_atom, func_d=func_d, solver_z=solver_z,
                reg=reg, n_iter=n_iter, random_state=0, verbose=0,
                sample_weights=None, ds_init=ds_init)

            assert_allclose(pobj_0, pobj_1)

    # sample_weights equal to 2 is equivalent to having twice the samples.
    # (with the regularization equal to zero)
    reg = 0.
    n_iter = 3
    n_duplicated = n_trials // 3
    sample_weights = np.ones_like(X)
    sample_weights[:n_duplicated] = 2
    X_duplicated = np.vstack([X[:n_duplicated], X])
    for func_d in [update_d, update_d_block]:
        for solver_z in ('l_bfgs', 'ista', 'fista'):
            pobj_0, _, d_hat_0, z_hat_0 = learn_d_z(
                X, n_atoms, n_times_atom, func_d=func_d, solver_z=solver_z,
                reg=reg, n_iter=n_iter, random_state=0, verbose=0,
                sample_weights=sample_weights, ds_init=ds_init,
                solver_z_kwargs=dict(factr=1e9))
            pobj_1, _, d_hat_1, z_hat_1 = learn_d_z(
                X_duplicated, n_atoms, n_times_atom, func_d=func_d,
                solver_z=solver_z, reg=reg, n_iter=n_iter, random_state=0,
                verbose=0, sample_weights=None, ds_init=ds_init,
                solver_z_kwargs=dict(factr=1e9))

            assert_allclose(pobj_0, pobj_1, rtol=0, atol=1e-3)
