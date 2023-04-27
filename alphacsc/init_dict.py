# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>
import numpy as np

from .update_d_multi import prox_uv, prox_d

from .utils.dictionary import tukey_window
from .utils.dictionary import get_uv
from .utils.validation import check_random_state


def get_init_strategy(n_times_atom, shape, random_state, D_init):
    """Returns dictionary initialization strategy.

    Parameters
    ----------
    n_times_atom : int
        The support of the atom.
    shape: tuple
        Expected shape of the dictionary. (n_atoms, n_channels + n_times_atoms)
    or (n_atoms, n_channels, n_times_atom)
    random_state: int or np.random.RandomState
        A seed to generate a RandomState instance or the instance itself.
    D_init : str or array, shape (n_atoms, n_channels + n_times_atoms) or \
                           shape (n_atoms, n_channels, n_times_atom)
        The initial atoms or an initialization scheme in
        {'chunk' | 'random' | 'greedy'}.
    """
    if isinstance(D_init, np.ndarray):
        return IdentityStrategy(shape, D_init)
    elif D_init is None or D_init == 'random':
        return RandomStrategy(shape, random_state)
    elif D_init == 'chunk':
        return ChunkStrategy(n_times_atom, shape, random_state)
    elif D_init == 'greedy':
        return GreedyStrategy(shape, random_state)
    else:
        raise NotImplementedError('It is not possible to initialize uv'
                                  ' with parameter {}.'.format(D_init))


class IdentityStrategy():
    """A class that creates a dictionary from a specified array.

    Parameters
    ----------
    shape: tuple
        Expected shape of the dictionary. (n_atoms, n_channels + n_times_atoms)
    or (n_atoms, n_channels, n_times_atom)
    D_init : array of shape (n_atoms, n_channels + n_times_atoms) or \
                      shape (n_atoms, n_channels, n_times_atom)
    """

    def __init__(self, shape, D_init):

        assert shape == D_init.shape
        self.D_init = D_init

    def initialize(self, X):
        return self.D_init.copy()


class RandomStrategy():
    """A class that creates a random dictionary for a specified shape.

    Parameters
    ----------
    shape: tuple
        Expected shape of the dictionary. (n_atoms, n_channels + n_times_atoms)
    or (n_atoms, n_channels, n_times_atom)
    random_state: int or np.random.RandomState
        A seed to generate a RandomState instance or the instance itself.
    """

    def __init__(self, shape, random_state):

        self.shape = shape
        self.random_state = random_state

    def initialize(self, X):
        rng = check_random_state(self.random_state)
        return rng.randn(*self.shape)


class ChunkStrategy():
    """A class that creates a random dictionary for a specified shape with
    'chunk' strategy.

    Parameters
    ----------
    n_times_atom : int
        The support of the atom.
    shape: tuple
        Expected shape of the dictionary. (n_atoms, n_channels + n_times_atoms)
    or (n_atoms, n_channels, n_times_atom)
    random_state: int or np.random.RandomState
        A seed to generate a RandomState instance or the instance itself.
    """

    def __init__(self, n_times_atom, shape, random_state):

        self.n_atoms = shape[0]
        self.n_times_atom = n_times_atom
        self.rank1 = True if len(shape) == 2 else False
        self.random_state = random_state

    def initialize(self, X):
        rng = check_random_state(self.random_state)
        n_trials, n_channels, n_times = X.shape

        D_hat = np.zeros(
            (self.n_atoms, n_channels, self.n_times_atom))

        for i_atom in range(self.n_atoms):
            i_trial = rng.randint(n_trials)
            t0 = rng.randint(n_times - self.n_times_atom)
            D_hat[i_atom] = X[i_trial, :, t0:t0 + self.n_times_atom].copy()

        if self.rank1:
            D_hat = get_uv(D_hat)

        return D_hat


class GreedyStrategy(RandomStrategy):
    """A class that creates a random dictionary for a specified shape and
    removes all elements.

    Parameters
    ----------
    shape: tuple
        Expected shape of the dictionary. (n_atoms, n_channels + n_times_atoms)
    or (n_atoms, n_channels, n_times_atom)
    random_state: int or np.random.RandomState
        A seed to generate a RandomState instance or the instance itself.
    """

    def initialize(self, X):
        D_hat = super().initialize(X)
        return D_hat[:0]


def init_dictionary(X, n_atoms, n_times_atom, uv_constraint='separate',
                    rank1=True, window=False, D_init=None, random_state=None):
    """Return an initial dictionary for the signals X

    Parameter
    ---------
    X: array, shape(n_trials, n_channels, n_times)
        The data on which to perform CSC.
    n_atoms: int
        The number of atoms to learn.
    n_times_atom: int
        The support of the atom.
    uv_constraint: str in {'joint' | 'separate'}
        The kind of norm constraint on the atoms:
        If 'joint', the constraint is norm_2([u, v]) <= 1
        If 'separate', the constraint is norm_2(u) <= 1 and norm_2(v) <= 1
    rank1: boolean
        If set to True, use a rank 1 dictionary.
    window: boolean
        If True, multiply the atoms with a temporal Tukey window.
    D_init: array or {'chunk' | 'random'}
        The initialization scheme for the dictionary or the initial
        atoms. The shape should match the required dictionary shape, ie if
        rank1 is True, (n_atoms, n_channels + n_times_atom) and else
        (n_atoms, n_channels, n_times_atom)
    random_state: int | None
        The random state.

    Return
    ------
    D: array shape(n_atoms, n_channels + n_times_atom) or
              shape(n_atoms, n_channels, n_times_atom)
        The initial atoms to learn from the data.
    """
    n_trials, n_channels, n_times = X.shape
    rng = check_random_state(random_state)

    D_shape = (n_atoms, n_channels, n_times_atom)
    if rank1:
        D_shape = (n_atoms, n_channels + n_times_atom)

    if isinstance(D_init, np.ndarray):
        D_hat = D_init.copy()
        assert D_hat.shape == D_shape

    elif D_init is None or D_init == "random":
        D_hat = rng.randn(*D_shape)

    elif D_init == 'chunk':
        D_hat = np.zeros((n_atoms, n_channels, n_times_atom))
        for i_atom in range(n_atoms):
            i_trial = rng.randint(n_trials)
            t0 = rng.randint(n_times - n_times_atom)
            D_hat[i_atom] = X[i_trial, :, t0:t0 + n_times_atom].copy()
        if rank1:
            D_hat = get_uv(D_hat)

    elif D_init == 'greedy':
        raise NotImplementedError()

    else:
        raise NotImplementedError('It is not possible to initialize uv with'
                                  ' parameter {}.'.format(D_init))

    if window and not isinstance(D_init, np.ndarray):
        if rank1:
            D_hat[:, n_channels:] *= tukey_window(n_times_atom)[None, :]
        else:
            D_hat = D_hat * tukey_window(n_times_atom)[None, None, :]

    if rank1:
        D_hat = prox_uv(D_hat, uv_constraint=uv_constraint,
                        n_channels=n_channels)
    else:
        D_hat = prox_d(D_hat)
    return D_hat
