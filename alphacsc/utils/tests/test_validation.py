import numpy as np
import pytest

from alphacsc.utils.validation import check_dimension


def test_check_dimension():
    X = np.empty((2, 3, 4))
    dims = check_dimension(X, expected_shape="n_trials, n_channels, n_times")
    assert dims == X.shape
    dims = check_dimension(X[0], expected_shape="n_trials, n_channels")
    assert dims == X.shape[1:]
    with pytest.raises(ValueError):
        check_dimension(X, expected_shape="n_trials, n_channels")
