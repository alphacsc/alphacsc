import numpy as np

import kmc2
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from .utils import check_random_state, _get_uv
from .update_d_multi import prox_uv
from .other.k_medoids import KMedoids


def init_uv(X, n_atoms, n_times_atom, uv_init=None, uv_constraint='separate',
            random_state=None, kmeans_params=dict()):
    """Return an initial dictionary for the signals X

    Parameter
    ---------
    X: array, shape (n_trials, n_channels, n_times)
        The data on which to perform CSC.
    n_atoms : int
        The number of atoms to learn.
    n_times_atom : int
        The support of the atom.
    uv_init : array, shape (n_atoms, n_channels + n_times_atoms)
        The initial atoms.
    uv_constraint : str in {'joint', 'separate', 'box'}
        The kind of norm constraint on the atoms:
        If 'joint', the constraint is norm_2([u, v]) <= 1
        If 'separate', the constraint is norm_2(u) <= 1 and norm_2(v) <= 1
        If 'box', the constraint is norm_inf([u, v]) <= 1
    random_state : int | None
        The random state.
    kmeans_params : dict
        Dictionnary of parameters for the kmeans init method.

    Return
    ------
    uv: array shape (n_atoms, n_channels + n_times_atom)
        The initial atoms to learn from the data.
    """
    n_trials, n_channels, n_times = X.shape
    rng = check_random_state(random_state)

    if isinstance(uv_init, np.ndarray):
        uv_hat = uv_init.copy()
        assert uv_hat.shape == (n_atoms, n_channels + n_times_atom)

    elif uv_init is None or uv_init == "random":
        uv_hat = rng.randn(n_atoms, n_channels + n_times_atom)

    elif uv_init == 'chunk':
        D_hat = np.zeros((n_atoms, n_channels, n_times_atom))
        for i_atom in range(n_atoms):
            i_trial = rng.randint(n_trials)
            t0 = rng.randint(n_times - n_times_atom)
            D_hat[i_atom] = X[i_trial, :, t0:t0 + n_times_atom]
        uv_hat = _get_uv(D_hat)

    elif uv_init == "kmeans":
        u_hat = rng.randn(n_atoms, n_channels)
        v_hat = kmeans_init(X, n_atoms, n_times_atom, random_state=rng,
                            **kmeans_params)
        uv_hat = np.c_[u_hat, v_hat]

    elif uv_init == "ssa":
        u_hat = rng.randn(n_atoms, n_channels)
        v_hat = ssa_init(X, n_atoms, n_times_atom, random_state=rng,
                         **kmeans_params)
        uv_hat = np.c_[u_hat, v_hat]
    elif uv_init == 'greedy':
        raise NotImplementedError()
    else:
        raise NotImplementedError('It is not possible to initialize uv with'
                                  ' parameter {}.'.format(uv_init))

    uv_hat = prox_uv(uv_hat, uv_constraint=uv_constraint, n_chan=n_channels)
    return uv_hat


def kmeans_init(X, n_atoms, n_times_atom, max_iter=0, random_state=None,
                non_uniform=True, use_custom_distances=False):
    """Return an initial temporal dictionary for the signals X

    Parameter
    ---------
    X: array, shape (n_trials, n_channels, n_times)
        The data on which to perform CSC.
    n_atoms : int
        The number of atoms to learn.
    n_times_atom : int
        The support of the atom.
    max_iter : int
        Number of iteration of kmeans algorithm
    random_state : int | None
        The random state.
    non_uniform : boolean
        If True, the kmc2 init uses the norm of each data chunk.
    use_custom_distances : boolean
        If True, the kmc2 init and the kmeans algorithm use a convolutional
        distance instead of the euclidean distance

    Return
    ------
    uv: array shape (n_atoms, n_channels + n_times_atom)
        The initial atoms to learn from the data.
    """
    if use_custom_distances:
        # Only take the strongest channels, otherwise X is too big
        n_strong_channels = 3
        strongest_channels = np.argsort(X.std(axis=2).mean(axis=0))
        X = X[:, strongest_channels[-n_strong_channels:], :]

    X = X.reshape(-1, X.shape[-1])

    # Time step between two windows
    step = max(1, n_times_atom // 3)

    # embed all the windows of length n_times_atom in X
    X_embed = np.concatenate(
        [_embed(Xi, n_times_atom).T[::step, :] for Xi in X])
    X_embed = np.atleast_2d(X_embed)

    if non_uniform:
        weights = np.linalg.norm(X_embed, axis=1)
    else:
        weights = None

    # init the kmeans centers with KMC2
    seeding, indices = kmc2.kmc2(X_embed, k=n_atoms, weights=weights,
                                 random_state=random_state,
                                 use_custom_distances=use_custom_distances)

    # perform the kmeans, or use the seeding if max_iter == 0
    if max_iter == 0:
        v_init = seeding

    elif use_custom_distances:
        model = KMedoids(n_clusters=n_atoms, init=np.int_(indices),
                         max_iter=max_iter,
                         random_state=random_state).fit(X_embed)
        v_init = model.cluster_centers_

    else:
        model = MiniBatchKMeans(n_clusters=n_atoms, init=seeding, n_init=1,
                                max_iter=max_iter,
                                random_state=random_state).fit(X_embed)
        v_init = model.cluster_centers_

    return v_init


def ssa_init(X, n_atoms, n_times_atom, random_state=None):
    """Return an initial temporal dictionary for the signals X

    Parameter
    ---------
    X: array, shape (n_trials, n_channels, n_times)
        The data on which to perform CSC.
    n_atoms : int
        The number of atoms to learn.
    n_times_atom : int
        The support of the atom.
    random_state : int | None
        The random state.

    Return
    ------
    uv: array shape (n_atoms, n_channels + n_times_atom)
        The initial atoms to learn from the data.
    """
    # Only take the strongest channel, otherwise X is too big
    strongest_channel = np.argmax(X.std(axis=2).mean(axis=0))
    X_strong = X[:, strongest_channel, :]

    # Time step between two windows
    step = 1

    # embed all the windows of length n_times_atom in X_strong
    X_embed = np.concatenate(
        [_embed(Xi, n_times_atom).T[::step, :] for Xi in X_strong])
    X_embed = np.atleast_2d(X_embed)

    model = PCA(n_components=n_atoms, random_state=random_state).fit(X_embed)
    v_init = model.components_

    return v_init


def _embed(x, dim, lag=1):
    """Create an embedding of array given a resulting dimension and lag.
    """
    x = x.copy()
    X = np.lib.stride_tricks.as_strided(x, (len(x) - dim * lag + lag, dim),
                                        (x.strides[0], x.strides[0] * lag))
    return X.T
