# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>
import itertools

import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .utils import check_random_state
from .other.kmc2 import custom_distances
from .update_d_multi import prox_uv, prox_d

from .utils.dictionary import tukey_window
from .utils.dictionary import get_uv, get_D
from .utils.dictionary import NoWindow, UVWindower, SimpleWindower


ried = custom_distances.roll_invariant_euclidean_distances
tied = custom_distances.translation_invariant_euclidean_distances


class BaseDictionary():

    def __init__(self, n_channels, n_atoms, n_times_atom, random_state,
                 window):
        self.n_channels = n_channels
        self.n_atoms = n_atoms
        self.n_times_atom = n_times_atom
        self.rng = check_random_state(random_state)

        if not window:
            self.windower = NoWindow()
        else:
            self._init_windower()

    def _init_windower(self):
        raise NotImplementedError()

    def window(self, D_hat):
        return self.windower.window(D_hat)

    def dewindow(self, D_hat):
        return self.windower.dewindow(D_hat)

    def simple_window(self, D_hat):
        return self.windower.simple_window(D_hat)

    def simple_dewindow(self, D_hat):
        return self.windower.simple_dewindow(D_hat)

    def prox(self, D_hat):
        raise NotImplementedError()

    def get_D_shape(self):
        raise NotImplementedError()

    def get_dict(self, X, D_init_params):
        D_hat = self.strategy.get_dict(X, D_init_params)

        if not hasattr(self.strategy, 'D_init'):
            D_hat = self.window(D_hat)
        D_hat = self.prox(D_hat)
        return D_hat

    def wrap(self, D_hat):
        return D_hat

    def wrap_rank1(self, D_hat):
        return D_hat

    def set_strategy(self, D_init):
        if isinstance(D_init, np.ndarray):
            self.strategy = IdentityStrategy(self, D_init)
        elif D_init is None or D_init == 'random':
            self.strategy = RandomStrategy(self)
        elif D_init == 'chunk':
            self.strategy = ChunkStrategy(self)
        elif D_init == "kmeans":
            self.strategy = KMeansStrategy(self)
        elif D_init == 'ssa':
            self.strategy = SSAStrategy(self)
        elif D_init == 'greedy':
            raise NotImplementedError()
        else:
            raise NotImplementedError('It is not possible to initialize uv'
                                      ' with parameter {}.'.format(D_init))


class Dictionary(BaseDictionary):

    def _init_windower(self):
        self.windower = SimpleWindower(self.n_times_atom)

    def prox(self, D_hat):
        return prox_d(D_hat)

    def get_D_shape(self):
        return (self.n_atoms, self.n_channels, self.n_times_atom)

    def wrap(self, D_hat):
        return get_D(D_hat, self.n_channels)


class Rank1Dictionary(BaseDictionary):

    def __init__(self, n_channels, n_atoms, n_times_atom, random_state,
                 window, uv_constraint):

        super().__init__(n_channels, n_atoms, n_times_atom, random_state,
                         window)

        self.uv_constraint = uv_constraint

    def _init_windower(self):
        self.windower = UVWindower(self.n_times_atom, self.n_channels)

    def prox(self, D_hat):
        return prox_uv(D_hat, uv_constraint=self.uv_constraint,
                       n_channels=self.n_channels)

    def get_D_shape(self):
        return (self.n_atoms, self.n_channels + self.n_times_atom)

    def wrap_rank1(self, D_hat):
        return get_uv(D_hat)


class BaseStrategy():

    def __init__(self, generator):
        self.generator = generator
        self.n_channels = generator.n_channels
        self.n_atoms = generator.n_atoms
        self.n_times_atom = generator.n_times_atom
        self.rng = generator.rng

    def get_D_shape(self):
        return self.generator.get_D_shape()

    def get_dict(self, X, D_init_params):
        raise NotImplementedError()

    def wrap(self, D_hat):
        return self.generator.wrap(D_hat)

    def wrap_rank1(self, D_hat):
        return self.generator.wrap_rank1(D_hat)


class IdentityStrategy(BaseStrategy):

    def __init__(self, generator, D_init):
        super().__init__(generator)

        assert self.get_D_shape() == D_init.shape
        self.D_init = D_init

    def get_dict(self, X, D_init_params):
        return self.D_init.copy()


class RandomStrategy(BaseStrategy):

    def get_dict(self, X, D_init_params):
        return self.rng.randn(*self.get_D_shape())


class ChunkStrategy(BaseStrategy):

    def get_dict(self, X, D_init_params):
        n_trials, n_channels, n_times = X.shape

        D_hat = np.zeros(
            (self.n_atoms, n_channels, self.n_times_atom))

        for i_atom in range(self.n_atoms):
            i_trial = self.rng.randint(n_trials)
            t0 = self.rng.randint(n_times - self.n_times_atom)
            D_hat[i_atom] = X[i_trial, :, t0:t0 + self.n_times_atom]

        D_hat = self.wrap_rank1(D_hat)
        return D_hat


class KMeansStrategy(BaseStrategy):

    def get_dict(self, X, D_init_params):
        D_hat = kmeans_init(X, self.n_atoms, self.n_times_atom,
                            random_state=self.rng, **D_init_params)

        D_hat = self.wrap(D_hat)
        return D_hat


class SSAStrategy(BaseStrategy):

    def get_dict(self, X, D_init_params):
        u_hat = self.rng.randn(self.n_atoms, self.n_channels)
        v_hat = ssa_init(X, self.n_atoms, self.n_times_atom,
                         random_state=self.rng)
        D_hat = np.c_[u_hat, v_hat]

        D_hat = self.wrap(D_hat)
        return D_hat


def init_dictionary(X, n_atoms, n_times_atom, uv_constraint='separate',
                    rank1=True, window=False, D_init=None,
                    D_init_params=dict(), random_state=None):
    """Return an initial dictionary for the signals X

    Parameter
    ---------
    X: array, shape (n_trials, n_channels, n_times)
        The data on which to perform CSC.
    n_atoms : int
        The number of atoms to learn.
    n_times_atom : int
        The support of the atom.
    uv_constraint : str in {'joint' | 'separate'}
        The kind of norm constraint on the atoms:
        If 'joint', the constraint is norm_2([u, v]) <= 1
        If 'separate', the constraint is norm_2(u) <= 1 and norm_2(v) <= 1
    rank1 : boolean
        If set to True, use a rank 1 dictionary.
    window : boolean
        If True, multiply the atoms with a temporal Tukey window.
    D_init : array or {'kmeans' | 'ssa' | 'chunk' | 'random'}
        The initialization scheme for the dictionary or the initial
        atoms. The shape should match the required dictionary shape, ie if
        rank1 is True, (n_atoms, n_channels + n_times_atom) and else
        (n_atoms, n_channels, n_times_atom)
    D_init_params : dict
        Dictionnary of parameters for the kmeans init method.
    random_state : int | None
        The random state.

    Return
    ------
    D : array shape (n_atoms, n_channels + n_times_atom) or
              shape (n_atoms, n_channels, n_times_atom)
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
            D_hat[i_atom] = X[i_trial, :, t0:t0 + n_times_atom]
        if rank1:
            D_hat = get_uv(D_hat)

    elif D_init == "kmeans":
        D_hat = kmeans_init(X, n_atoms, n_times_atom, random_state=rng,
                            **D_init_params)
        if not rank1:
            D_hat = get_D(D_hat, n_channels)

    elif D_init == "ssa":
        u_hat = rng.randn(n_atoms, n_channels)
        v_hat = ssa_init(X, n_atoms, n_times_atom, random_state=rng)
        D_hat = np.c_[u_hat, v_hat]
        if not rank1:
            D_hat = get_D(D_hat, n_channels)

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
    rng = check_random_state(random_state)

    n_trials, n_channels, n_times = X.shape
    X_original = X
    if distances != 'euclidean':
        # Only take the strongest channels, otherwise X is too big
        n_strong_channels = 1
        strongest_channels = np.argsort(X.std(axis=2).mean(axis=0))
        X = X[:, strongest_channels[-n_strong_channels:], :]

    X = X.reshape(-1, X.shape[-1])
    n_trials, n_times = X.shape

    # Time step between two windows
    step = max(1, n_times_atom // 3)

    # embed all windows of length n_times_atom in X
    X_embed = np.concatenate(
        [_embed(Xi, n_times_atom).T[::step, :] for Xi in X])
    X_embed = np.atleast_2d(X_embed)

    if non_uniform:
        weights = np.linalg.norm(X_embed, axis=1)
    else:
        weights = None

    # init the kmeans centers with KMC2
    try:
        from alphacsc.other.kmc2 import kmc2
        seeding, indices = kmc2.kmc2(X_embed, k=n_atoms, weights=weights,
                                     random_state=rng, distances=distances)
    except ImportError:
        if max_iter == 0:
            raise ImportError("Could not import alphacsc.other.kmc2. This "
                              "breaks the logic for the D_init='kmeans'. It "
                              "should not be used with max_iter=0 in "
                              "D_init_params.")
        # Default to random init for non-euclidean distances and to "kmeans++"
        # in the case of K-means.
        indices = rng.choice(len(X_embed), size=n_atoms, replace=False)
        seeding = "kmeans++"

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

        try:
            from .other.k_medoids import KMedoids
        except ImportError:
            raise ImportError("Could not import multics.other.k_medoid, make "
                              "sure to compile it to be able to initialize "
                              "the dictionary with k-means and a non-euclidean"
                              " distance.")
        model = KMedoids(n_clusters=n_atoms, init=np.int_(indices),
                         max_iter=max_iter, distance_metric=distance_metric,
                         random_state=rng).fit(X_embed)
        indices = model.medoid_idxs_
        labels = model.labels_

    else:
        distance_metric = 'euclidean'
        model = MiniBatchKMeans(n_clusters=n_atoms, init=seeding, n_init=1,
                                max_iter=max_iter, random_state=rng
                                ).fit(X_embed)
        v_init = model.cluster_centers_
        u_init = rng.randn(n_atoms, n_channels)
        D_init = np.c_[u_init, v_init]
        labels = model.labels_

    if tsne:
        if distances == 'euclidean':
            X_embed = X_embed[::100]
            if labels is not None:
                labels = labels[::100]
        plot_tsne(X_embed, v_init, labels=labels, metric=distance_metric,
                  random_state=rng)

    if not (distances == 'euclidean' and max_iter > 0):
        indices = np.array(indices)
        n_window = X_embed.shape[0] // n_trials
        medoid_i = (indices // n_window) // n_channels
        medoid_t = (indices % n_window) * step
        D = np.array([X_original[i, :, t:t + n_times_atom]
                      for i, t in zip(medoid_i, medoid_t)])
        D_init = get_uv(D)

    return D_init


def plot_tsne(X_embed, X_centers, labels=None, metric='euclidean',
              random_state=None):

    import matplotlib.pyplot as plt
    from .viz.callback import COLORS

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=5,
                metric=metric, verbose=2)
    pca = PCA(n_components=min(10, X_embed.shape[1]))
    X = np.concatenate([X_embed, X_centers])
    n_centers = X_centers.shape[0]
    X_pca = pca.fit_transform(X)
    X_tsne = tsne.fit_transform(X_pca)

    if labels is not None:
        labels = np.r_[labels, np.arange(n_centers)]
        colors = [c for c, l in zip(itertools.cycle(COLORS),
                                    np.unique(labels))]
        colors = np.array(colors)[labels]
    else:
        colors = None

    plt.figure('tsne')
    cc = colors[:-n_centers] if colors is not None else None
    plt.scatter(X_tsne[:-n_centers, 0], X_tsne[:-n_centers, 1], c=cc,
                marker='.')
    cc = colors[-n_centers:] if colors is not None else None
    plt.scatter(X_tsne[-n_centers:, 0], X_tsne[-n_centers:, 1], c=cc,
                marker='*', s=4)


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
