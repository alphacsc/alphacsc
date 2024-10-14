import numpy as np
from scipy.signal import hilbert
from scipy.signal.windows import tukey


def fast_hilbert(array):
    n_points = array.shape[0]
    n_fft = next_power2(n_points)
    return hilbert(array, n_fft)[:n_points]


def next_power2(num):
    """Compute the smallest power of 2 >= to num.(float -> int)"""
    return 2 ** int(np.ceil(np.log2(num)))


def split_signal(X, n_splits=1, apply_window=True):
    """Split the signal in ``n_splits`` chunks for faster training.

    This function can be used to accelerate the dictionary learning algorithm
    by creating independent chunks that can be processed in parallel. This can
    bias the estimation and can create border artifacts so the number of chunks
    should be kept as small as possible (`e.g.` equal to ``n_jobs``).

    Also, it is advised to not use the result of this function to
    call the ``DictionaryLearning.transform`` method, as it would return an
    approximate reduction of the original signal in the sparse basis.

    Note that this is a lossy operation, as all chunks will have length
    ``n_times // n_splits`` and the last ``n_times % n_splits`` samples are
    discarded.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times) or (1, n_channels, n_times)
        Signal to be split. It should be a single signal.
    n_splits : int (default: 1)
        Number of splits to create from the original signal. Default is 1.
    apply_window : bool (default: True)
        If set to True (default), a tukey window is applied to each split to
        reduce the border artifacts by reducing the weights of the chunk
        borders.

    Returns
    -------
    X_split: ndarray, shape (n_splits, n_channels, n_times // n_splits)
        The signal splitted in ``n_splits``.
    """
    msg = "This splitting utility is only designed for one multivariate signal"
    if X.ndim == 3:
        assert X.shape[0] == 1, (
            msg + "(1, n_channels, n_times. Found X.shape={}".format(X.shape))
        X = X[0]
    assert X.ndim == 2, (
        msg + " (n_channels, n_times). Found X.ndim={}.".format(X.ndim))

    n_splits = int(n_splits)
    assert n_splits > 0, "The number of splits should be large than 0."

    n_channels, n_times = X.shape
    n_times = n_times // n_splits
    X_split = X[:, :n_splits * n_times].copy()
    X_split = X_split.reshape(n_channels, n_splits, n_times).swapaxes(0, 1)

    # Apply a window to the signal to reduce the border artifacts
    if apply_window:
        X_split *= tukey(n_times, alpha=0.1)[None, None, :]

    return X_split


def check_univariate_signal(X):
    """Return an array that can be used with alphacsc transformers for
    univariate signals.

    Parameters
    ----------
    X : ndarray, shape (n_times,) or (n_trials, n_times)
        Signal to be reshaped. It should be a single signal.

    Returns
    -------
    X: ndarray, shape (n_trials, n_channels, n_times)
        The signal with the correct number of dimensions to be use with
        alphacsc transformers.
    """
    if X.ndim == 1:
        return X.reshape(1, 1, -1)
    if X.ndim == 2:
        return X[:, None]
    raise ValueError("This utility should only be used for univariate signals "
                     "with shape (n_times,) or (n_trials, n_times). Got {}."
                     .format(X.shape))


def check_multivariate_signal(X):
    """Return an array that can be used with alphacsc transformers for
    multivariate signals.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times) or (n_trials, n_channels, n_times)
        Signal to be reshaped. It should be a single signal.

    Returns
    -------
    X: ndarray, shape (n_trials, n_channels, n_times)
        The signal with the correct number of dimensions to be use with
        alphacsc transformers.
    """
    if X.ndim == 2:
        return X[None]
    if X.ndim == 3:
        return X
    raise ValueError("This utility should only be used with multivariate "
                     "signals with shape (n_channels, n_times) or "
                     "(n_trials, n_channels, n_times). Got {}."
                     .format(X.shape))
