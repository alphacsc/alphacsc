import numpy as np
from numpy.testing import assert_array_almost_equal

from alphacsc.other.kmc2 import custom_distances

ried = custom_distances.roll_invariant_euclidean_distances
tied = custom_distances.translation_invariant_euclidean_distances


def test_roll_invariant_euclidean_distance():
    rng = np.random.RandomState(0)

    for n_features in range(50, 60):
        X = rng.randn(2, n_features)
        Y = rng.randn(2, n_features)
        distances = ried(X, Y)
        for ii in range(X.shape[0]):
            for jj in range(Y.shape[0]):
                reference = min([
                    np.power(np.roll(X[ii], kk) - Y[jj], 2).sum()
                    for kk in range(n_features)
                ])
                assert_array_almost_equal(distances[ii, jj], reference)


def test_translation_invariant_euclidean_distance_symmetric():
    rng = np.random.RandomState(0)

    n_samples, n_features = 3, 26
    X = rng.randn(n_samples, 2 * n_features)
    Y = rng.randn(n_samples + 1, 2 * n_features)

    distances = tied(X, Y, symmetric=True)

    for ii in range(X.shape[0]):
        for jj in range(Y.shape[0]):
            reference = min([
                np.power(X[ii, kk:kk + n_features] - Y[jj, ll:ll + n_features],
                         2).sum()
                for kk in range(n_features) for ll in range(n_features)
            ])
            assert_array_almost_equal(distances[ii, jj], reference)


def test_translation_invariant_euclidean_distance_asymmetric():
    rng = np.random.RandomState(0)

    n_samples, n_features = 3, 26
    X = rng.randn(n_samples, 2 * n_features)
    Y = rng.randn(n_samples + 1, 2 * n_features)

    distances = tied(X, Y, symmetric=False)

    ll = n_features // 2

    for ii in range(X.shape[0]):
        for jj in range(Y.shape[0]):
            reference = min([
                np.power(X[ii, kk:kk + n_features] - Y[jj, ll:ll + n_features],
                         2).sum() for kk in range(n_features)
            ])
            assert_array_almost_equal(distances[ii, jj], reference)
