import numpy as np
import pytest

from alphacsc.utils import check_dimension


def test_check_dimension():
    X = np.empty((2, 3, 4))
    check_dimension(X, expected_shape="n_trials, n_channels, n_times")
    check_dimension(X[0], expected_shape="n_trials, n_channels")
    with pytest.raises(ValueError):
        check_dimension(X, expected_shape="n_trials, n_channels")
