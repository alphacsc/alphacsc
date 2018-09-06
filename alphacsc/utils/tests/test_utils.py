import numpy as np
from scipy import sparse
from numpy.testing import assert_allclose

from alphacsc.utils.convolution import _sparse_convolve, _dense_convolve
from alphacsc.utils.convolution import _choose_convolve
from alphacsc.utils import check_random_state
from alphacsc.utils import construct_X_multi
from alphacsc.utils.dictionary import get_D, get_uv

from alphacsc.update_d_multi import prox_uv


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


def test_construct_X():
    rng = check_random_state(42)
    n_times_atoms, n_times = 21, 128
    n_atoms = 3
    n_trials, n_channels = 29, 7
    n_times_valid = n_times - n_times_atoms + 1
    density = 0.1
    zi = sparse.random(n_atoms * n_trials, n_times_valid, density,
                       random_state=rng).toarray().reshape(
                           (n_trials, n_atoms, n_times_valid))
    uv = rng.randn(n_atoms, n_channels + n_times_atoms)
    ds = get_D(uv, n_channels)

    X_uv = construct_X_multi(zi, D=uv, n_channels=n_channels)
    X_ds = construct_X_multi(zi, D=ds)

    assert_allclose(X_uv, X_ds, atol=1e-16)


def test_uv_D():
    rng = check_random_state(42)
    n_times_atoms = 21
    n_atoms = 15
    n_channels = 30

    uv = rng.randn(n_atoms, n_channels + n_times_atoms)
    uv = prox_uv(uv, uv_constraint='separate', n_channels=n_channels)
    ds = get_D(uv, n_channels)
    uv_hat = get_uv(ds)

    assert_allclose(abs(uv / uv_hat), 1)


def test_patch_reconstruction_error():
    rng = check_random_state(42)
    n_times_atoms, n_times = 21, 128
    n_atoms = 3
    n_trials, n_channels = 29, 7
    n_times_valid = n_times - n_times_atoms + 1
    density = 0.1
    z = sparse.random(n_atoms * n_trials, n_times_valid, density,
                      random_state=rng).toarray().reshape(
                          (n_trials, n_atoms, n_times_valid))
    uv = rng.randn(n_atoms, n_channels + n_times_atoms)

    X = construct_X_multi(z, D=uv, n_channels=n_channels)

    from alphacsc.utils.dictionary import _patch_reconstruction_error

    rec = _patch_reconstruction_error(X, z, uv)
    assert rec.shape == (n_trials, n_times_valid)
    assert_allclose(rec, 0)

    uv = rng.randn(n_atoms, n_channels + n_times_atoms)
    rec = _patch_reconstruction_error(X, z, uv)
    X_hat = construct_X_multi(z, D=uv, n_channels=n_channels)

    for i in range(10):
        for j in range(10):
            assert np.isclose(rec[i, j], np.sum(
                (X_hat[i, :, j:j + n_times_atoms] -
                 X[i, :, j:j + n_times_atoms])**2))
