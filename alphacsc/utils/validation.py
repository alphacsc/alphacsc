import numpy as np


def check_consistent_shape(*args):
    for array in args[1:]:
        if array is not None and array.shape != args[0].shape:
            raise ValueError('Incompatible shapes. Got '
                             '(%s != %s)' % (array.shape, args[0].shape))


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def check_dimension(X, expected_shape="n_trials, n_channels, n_times"):
    """Check the dimension of the input signal and return it.

    If the dimension is not correct, return a sensible error message.

    Parameters
    -----------
    X : ndarray
        Input signal
    expected_shape: str
        Expected shape of the input signal. If X is of lower dimensionality,
        raise a sensible error.

    Return
    ------
    shape : tuple
        The shape of the input signal.
    """
    ndim = len(expected_shape.split(','))
    if X.ndim != ndim:
        raise ValueError("Expected shape ({}) but got input signal with "
                         "ndim={} and shape {}".format(
                             expected_shape, X.ndim, X.shape))

    return X.shape
