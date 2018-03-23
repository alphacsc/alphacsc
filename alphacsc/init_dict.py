import numpy as np
import matplotlib.pyplot as plt

import kmc2
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .utils import check_random_state
from .utils.dictionary import _get_uv, _patch_reconstruction_error
from .update_d_multi import prox_uv
from .other.k_medoids import KMedoids
from .other.kmc2 import custom_distances

ried = custom_distances.roll_invariant_euclidean_distances
tied = custom_distances.translation_invariant_euclidean_distances


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
                non_uniform=True, distances='euclidean', tsne=False):
    """Return an initial temporal dictionary for the signals X

    Parameter
    ---------
    X : array, shape (n_trials, n_channels, n_times)
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
    distances : str in {'euclidean', 'roll_inv', 'trans_inv'}
        Distance kind.

    Return
    ------
    uv: array shape (n_atoms, n_channels + n_times_atom)
        The initial atoms to learn from the data.
    """
    n_trials, n_channels, n_times = X.shape
    if distances != 'euclidean':
        # Only take the strongest channels, otherwise X is too big
        n_strong_channels = 1
        strongest_channels = np.argsort(X.std(axis=2).mean(axis=0))
        X = X[:, strongest_channels[-n_strong_channels:], :]

    X = X.reshape(-1, X.shape[-1])

    # Time step between two windows
    step = max(1, n_times_atom)

    # embed all windows of length n_times_atom in X
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
                                 distances=distances)

    # perform the kmeans, or use the seeding if max_iter == 0
    if max_iter == 0:
        v_init = seeding
        labels = None
        distance_metric = 'euclidean'

    elif distances != 'euclidean':
        if distances == 'trans_inv':
            distance_metric = tied
        elif distances == 'roll_inv':
            distance_metric = ried
        else:
            raise ValueError('Unknown distance "%s".' % (distances, ))

        model = KMedoids(n_clusters=n_atoms, init=np.int_(indices),
                         max_iter=max_iter, distance_metric=distance_metric,
                         random_state=random_state).fit(X_embed)
        v_init = model.cluster_centers_
        labels = model.labels_

    else:
        distance_metric = 'euclidean'
        model = MiniBatchKMeans(n_clusters=n_atoms, init=seeding, n_init=1,
                                max_iter=max_iter,
                                random_state=random_state).fit(X_embed)
        v_init = model.cluster_centers_
        labels = model.labels_

    if tsne:
        if distances == 'euclidean':
            X_embed = X_embed[::n_channels]
            if labels is not None:
                labels = labels[::n_channels]
        plot_tsne(X_embed, v_init, labels=labels, metric=distance_metric,
                  random_state=random_state)

    return v_init


def plot_tsne(X_embed, X_centers, labels=None, metric='euclidean',
              random_state=None):

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=15,
                metric=metric, verbose=2)
    pca = PCA(n_components=min(50, X_embed.shape[1]))
    X = np.concatenate([X_embed, X_centers])
    n_centers = X_centers.shape[0]
    X_pca = pca.fit_transform(X)
    X_tsne = tsne.fit_transform(X_pca)

    if labels is not None:
        labels = np.r_[labels, np.arange(n_centers)]
        colors = [
            "#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"
        ]
        colors = np.array(colors)[labels]
    else:
        colors = None

    fig = plt.figure('tsne')
    cc = colors[:-n_centers] if colors is not None else None
    plt.scatter(X_tsne[:-n_centers, 0], X_tsne[:-n_centers, 1], c=cc,
                marker='.')
    cc = colors[-n_centers:] if colors is not None else None
    plt.scatter(X_tsne[-n_centers:, 0], X_tsne[-n_centers:, 1], c=cc,
                marker='*')


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

    # Time step between two windows
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


def get_max_error_dict(X, Z, uv):
    """Get the maximal reconstruction error patch from the data as a new atom

    This idea is used for instance in [Yellin2017]

    Parameters
    ----------
    X: array, shape (n_trials, n_channels, n_times)
        Signals encoded in the CSC.
    Z: array, shape (n_atoms, n_trials, n_times_valid)
        Current estimate of the coding signals.
    uv: array, shape (n_atoms, n_channels + n_times_atom)
        Current estimate of the rank1 multivariate dictionary.

    Return
    ------
    uvk: array, shape (n_channels + n_times_atom,)
        New atom for the dictionary, chosen as the chunk of data with the
        maximal reconstruction error.

    [Yellin2017] BLOOD CELL DETECTION AND COUNTING IN HOLOGRAPHIC LENS-FREE
    IMAGING BY CONVOLUTIONAL SPARSE DICTIONARY LEARNING AND CODING.
    """
    n_trials, n_channels, n_times = X.shape
    n_times_atom = uv.shape[1] - n_channels
    patch_rec_error = _patch_reconstruction_error(X, Z, uv)
    i0 = patch_rec_error.argmax()
    n0, t0 = np.unravel_index(i0, Z.shape[1:])

    d0 = X[n0, :, t0:t0 + n_times_atom][None]

    return _get_uv(d0)
