"""Convolutional dictionary learning"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>


from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError

from .update_z_multi import update_z_multi
from .utils.dictionary import get_D, get_uv
from .learn_d_z_multi import learn_d_z_multi


DOC_FMT = """{desc}

    Methods
    -------

    __init__: instantiate a class to perform CDL.
    fit : learn a convolutional dictionary from a given set of signal X
    transform : return the sparse codes associated to the learned dictionary

    Parameters
    ----------

    Problem Specs
    ~~~~~~~~~~~~~
    n_atoms : int
        The number of atoms to learn.
    n_times_atom : int
        The support of the atom.
    reg : float
        The regularization parameter
    loss : {{ 'l2' | 'dtw' | 'whitening' }}
        Loss for the data-fit term. Either the norm l2 or the soft-DTW.
    loss_params : dict
        Parameters of the loss
    rank1 : boolean
        If set to True, learn rank 1 dictionary atoms.
    uv_constraint : {{'joint', 'separate', 'box'}}
        The kind of norm constraint on the atoms:
        If 'joint', the constraint is norm_2([u, v]) <= 1
        If 'separate', the constraint is norm_2(u) <= 1 and norm_2(v) <= 1
        If 'box', the constraint is norm_inf([u, v]) <= 1


    Global algorithm
    ~~~~~~~~~~~~~~~~
    {algorithm}
    n_iter : int
        The number of alternate steps to perform.
    eps : float
        Stopping criterion. If the cost descent after a uv and a z update is
        smaller than eps, return.
    lmbd_max : {{'fixed' | 'scaled' | 'per_atom' | 'shared'}}
        If not fixed, adapt the regularization rate as a ratio of lambda_max.


    Z-step parameters
    ~~~~~~~~~~~~~~~~~
    solver_z : str
        The solver to use for the z update. Options are
        'l_bfgs' (default) | "lgcd"
    solver_z_kwargs : dict
        Additional keyword arguments to pass to update_z_multi


    D-step parameters
    ~~~~~~~~~~~~~~~~~
    solver_d : str
        The solver to use for the d update. Options are
        'alternate' | 'alternate_adaptive' (default) | 'joint' | 'l-bfgs'
    solver_d_kwargs : dict
        Additional keyword arguments to provide to update_d
    D_init : str or array, shape (n_atoms, n_channels + n_times_atoms) or
                            shape (n_atoms, n_channels, n_times_atom)
        The initial atoms or an initialization scheme in
        {{'kmeans' | 'ssa' | 'chunks' | 'random'}}.
    D_init_params : dict
        Dictionnary of parameters for the kmeans init method.
    use_sparse_z : boolean
        Use sparse lil_matrices to store the activations.


    Technical parameters
    ~~~~~~~~~~~~~~~~~~~~
    n_jobs : int
        The number of parallel jobs.
    verbose : int
        The verbosity level.
    callback : func
        A callback function called at the end of each loop of the
        coordinate descent.
    random_state : int | None
        The random state.
    raise_on_increase : boolean
        Raise an error if the objective function increase


    Attributes
    ----------
    D_hat_ : array, shape (n_atoms, n_channels, n_times_atom)
        The dictionary in full rank mode.
    uv_hat_ : array, shape (n_atoms, n_channels + n_times_atom)
        The dictionary in rank 1 mode. If `rank1 = False`, this is an
        approximation of the dictionary obtained through svd.
    u_hat_ : array, shape (n_atoms, n_channels)
        The spatial map of the dictionary in rank 1 mode. If `rank1 = False`,
        this is an approximation of the dictionary obtained through svd.
    v_hat_ : array, shape (n_atoms, n_times_atom)
        The temporal patterns of the dictionary in rank 1 mode. If
        `rank1 = False`, this is an approximation of the dictionary obtained
        through svd.
    z_hat_ : array, shape (n_trials, n_atoms, n_times_valid)
        The sparse code associated to the signals used to fit the model.
    pobj_ : list
        The objective function value at each step of the coordinate descent.
    times_ : list
        The cumulative time for each iteration of the coordinate descent.
    """

DEFAULT = dict(
    desc="Base class for convolutional dictionary learning algorithms",
    algorithm="""
    algorithm : {'batch' | 'greedy' | 'online'}
        Dictionary learning algorithm.
    algorithm_params : dict
        parameter of the global algorithm"""
)


class ConvolutionalDictionaryLearning(TransformerMixin):
    __doc__ = DOC_FMT.format(**DEFAULT)

    def __init__(self, n_atoms, n_times_atom, reg=0.1, n_iter=60, n_jobs=1,
                 loss='l2', loss_params=dict(gamma=.1, sakoe_chiba_band=10,
                                             ordar=10),
                 rank1=True, uv_constraint='separate',
                 solver_z='l_bfgs', solver_z_kwargs={},
                 solver_d='alternate_adaptive', solver_d_kwargs={},
                 eps=1e-10, D_init=None, D_init_params={},
                 algorithm='batch', algorithm_params={},
                 use_sparse_z=False, lmbd_max='fixed', verbose=10,
                 callback=None, random_state=None, name="_CDL",
                 alpha=.8, batch_size=1, batch_selection='random',
                 raise_on_increase=True):

        # Problem Specs
        self.n_atoms = n_atoms
        self.n_times_atom = n_times_atom
        self.reg = reg
        self.loss = loss
        self.loss_params = loss_params
        self.rank1 = rank1
        self.uv_constraint = uv_constraint

        # Global algorithm
        self.n_iter = n_iter
        self.eps = eps
        self.algorithm = algorithm
        self.algorithm_params = algorithm_params
        self.lmbd_max = lmbd_max

        # Z-step parameters
        self.solver_z = solver_z
        self.solver_z_kwargs = solver_z_kwargs

        # D-step parameters
        self.solver_d = solver_d
        self.solver_d_kwargs = solver_d_kwargs
        self.D_init = D_init
        self.D_init_params = D_init_params
        self.use_sparse_z = use_sparse_z

        # Technical parameters
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.callback = callback
        self.random_state = random_state
        self.raise_on_increase = raise_on_increase
        self.name = name

        # Init property
        self._D_hat = None

    def fit(self, X, y=None):
        self.pobj_, self.times_, self._D_hat, self.z_hat_ = learn_d_z_multi(
            X, self.n_atoms, self.n_times_atom, reg=self.reg,
            loss=self.loss, loss_params=self.loss_params,
            rank1=self.rank1, uv_constraint=self.uv_constraint,
            algorithm=self.algorithm, algorithm_params=self.algorithm_params,
            n_iter=self.n_iter, eps=self.eps,
            solver_z=self.solver_z, solver_z_kwargs=self.solver_z_kwargs,
            solver_d=self.solver_d, solver_d_kwargs=self.solver_d_kwargs,
            D_init=self.D_init, D_init_params=self.D_init_params,
            use_sparse_z=self.use_sparse_z, lmbd_max=self.lmbd_max,
            verbose=self.verbose, callback=self.callback,
            random_state=self.random_state, n_jobs=self.n_jobs,
            name=self.name, raise_on_increase=self.raise_on_increase)
        self.n_channels_ = X.shape[1]
        return self

    def transform(self, X):
        assert self._D_hat is not None
        z_hat, _, _ = update_z_multi(
            X, self._D_hat, reg=self.reg, z0=self.z0, n_jobs=self.n_jobs,
            solver=self.solver, solver_kwargs=self.solver_z_kwargs,
            loss=self.loss, loss_params=self.loss_params)

    def _check_fitted(self):
        if self._D_hat is None:
            raise NotFittedError("Fit must be called before accessing the "
                                 "dictionary")

    @property
    def D_hat_(self):
        self._check_fitted()
        if self._D_hat.ndim == 3:
            return self._D_hat

        return get_D(self._D_hat, self.n_channels_)

    @property
    def uv_hat_(self):
        self._check_fitted()
        if self._D_hat.ndim == 3:
            return get_uv(self._D_hat)

        return self._D_hat

    @property
    def u_hat_(self):
        return self.uv_hat_[:, :self.n_channels_]

    @property
    def v_hat_(self):
        return self.uv_hat_[:, self.n_channels_:]


class BatchCDL(ConvolutionalDictionaryLearning):
    _default = {}
    _default.update(DEFAULT)
    _default['desc'] = "Batch algorithm for convolutional dictionary learning"
    _default['algorithm'] = "Batch algorithm"
    __doc__ = DOC_FMT.format(**_default)

    def __init__(self, n_atoms, n_times_atom, reg=0.1, n_iter=60, n_jobs=1,
                 solver_z='lgcd', solver_z_kwargs={},
                 solver_d='alternate_adaptive', solver_d_kwargs={},
                 rank1=True, uv_constraint='separate', lmbd_max='scaled',
                 eps=1e-10, D_init=None, D_init_params={},
                 verbose=10, random_state=None):
        super().__init__(
            n_atoms, n_times_atom, reg=reg, n_iter=n_iter,
            solver_z=solver_z, solver_z_kwargs=solver_z_kwargs,
            solver_d=solver_d, solver_d_kwargs=solver_d_kwargs,
            rank1=rank1, uv_constraint=uv_constraint,
            eps=eps, D_init=D_init, D_init_params=D_init_params,
            algorithm='batch', lmbd_max=lmbd_max, raise_on_increase=True,
            loss='l2', use_sparse_z=False, n_jobs=n_jobs, verbose=verbose,
            callback=None, random_state=random_state, name="BatchCDL")


class OnlineCDL(ConvolutionalDictionaryLearning):
    _default = {}
    _default.update(DEFAULT)
    _default['desc'] = "Online algorithm for convolutional dictionary learning"
    _default['algorithm'] = """Online algorithm
    alpha : float
        Forgetting factor for online learning. If set to 0, the learning is
        stochastic and each D-step is independent from the previous steps.
        When set to 1, each the previous values z_hat - computed with different
        dictionary - have the same weight as the current one. This factor
        should be large enough to ensure convergence but to large factor can
        lead to sub-optimal minima.
    batch_selection : 'random' | 'cyclic'
        The batch selection strategy for online learning. The batch are either
        selected randomly among all samples (without replacement) or in a
        cyclic way.
    batch_size : int in [1, n_trials]
        Size of the batch used in online learning. Increasing it regularizes
        the dictionary learning as there is less variance for the successive
        estimates. But it also increases the computational cost as more coding
        signals z_hat must be estimate at each iteration."""
    __doc__ = DOC_FMT.format(**_default)

    def __init__(self, n_atoms, n_times_atom, reg=0.1, n_iter=60, n_jobs=1,
                 solver_z='lgcd', solver_z_kwargs={},
                 solver_d='alternate_adaptive', solver_d_kwargs={},
                 rank1=True, uv_constraint='separate', lmbd_max='scaled',
                 eps=1e-10, D_init=None, D_init_params={},
                 alpha=.8, batch_size=1, batch_selection='random',
                 verbose=10, random_state=None):
        super().__init__(
            n_atoms, n_times_atom, reg=reg, n_iter=n_iter,
            solver_z=solver_z, solver_z_kwargs=solver_z_kwargs,
            solver_d=solver_d, solver_d_kwargs=solver_d_kwargs,
            rank1=rank1, uv_constraint=uv_constraint,
            eps=eps, D_init=D_init, D_init_params=D_init_params,
            algorithm_params=dict(alpha=alpha, batch_size=batch_size,
                                  batch_selection=batch_selection),
            n_jobs=n_jobs, random_state=random_state, algorithm='online',
            lmbd_max=lmbd_max, raise_on_increase=False, loss='l2',
            callback=None, use_sparse_z=False, verbose=verbose,
            name="OnlineCDL")
