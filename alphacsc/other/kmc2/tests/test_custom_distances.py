import numpy as np
from numpy.testing import assert_array_almost_equal

from alphacsc.other.kmc2.custom_distances import \
    roll_invariant_euclidean_distances


def test_roll_invariant_euclidean_distance():
    rng = np.random.RandomState(0)

    for n_features in range(100, 110):
        X = rng.randn(n_features)
        Y = rng.randn(n_features)
        distances = roll_invariant_euclidean_distances(X[None, :], Y[None, :])
        reference = min(
            [np.power(np.roll(X, i) - Y, 2).sum() for i in range(n_features)])
        assert_array_almost_equal(distances[0, 0], reference)
