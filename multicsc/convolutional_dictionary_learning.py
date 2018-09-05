"""Convolutional dictionary learning"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>


from sklearn.base import TransformerMixin
from .update_z_multi import update_z_multi
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
    lmbd_max : {{'fixed' | 'per_atom' | 'shared'}}
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


    Returns
    -------
    pobj : list
        The objective function value at each step of the coordinate descent.
    times : list
        The cumulative time for each iteration of the coordinate descent.
    uv_hat : array, shape (n_atoms, n_channels + n_times_atom)
        The atoms to learn from the data.
    z_hat : array, shape (n_trials, n_atoms, n_times_valid)
        The sparse activation matrix.
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
        self.D_hat = None

    def fit(self, X, y=None):
        self.pobj_, self.times_, self.D_hat, self.z_hat = learn_d_z_multi(
            X, self.n_atoms, self.n_times_atom, reg=self.reg,
            loss=self.loss, loss_params=self.loss_params,
            rank1=self.rank1, uv_constraint=self.uv_constraint,
            algorithm=self.algorithm, algorithm_params=self.algorithm_params,
            n_iter=self.n_iter, eps=self.eps, stopping_pobj=self.stopping_pobj,
            solver_z=self.solver_z, solver_z_kwargs=self.solver_z_kwargs,
            solver_d=self.solver_d, solver_d_kwargs=self.solver_d_kwargs,
            D_init=self.D_init, D_init_params=self.D_init_params,
            use_sparse_z=self.use_sparse_z, lmbd_max=self.lmbd_max,
            verbose=self.verbose, callback=self.callback,
            random_state=self.random_state, n_jobs=self.n_jobs,
            name=self.name, raise_on_increase=True)

    def transform(self, X):
        assert self.D_hat is not None
        z_hat, _, _ = update_z_multi(
            X, self.D_hat, reg=self.reg, z0=self.z0, parallel=self.parallel,
            solver=self.solver, solver_kwargs=self.solver_z_kwargs,
            loss=self.loss, loss_params=self.loss_params)


class BatchCDL(ConvolutionalDictionaryLearning):
    _default = {}
    _default.update(DEFAULT)
    _default['desc'] = "Batch algorithm for convolutional dictionary learning"
    _default['algorithm'] = "Batch algorithm"
    __doc__ = DOC_FMT.format(**_default)

    def __init__(self, n_atoms, n_times_atom, reg=0.1, n_iter=60, n_jobs=1,
                 solver_z='lgcd', solver_z_kwargs={},
                 solver_d='alternate_adaptive', solver_d_kwargs={},
                 rank1=True, uv_constraint='separate',
                 eps=1e-10, D_init=None, D_init_params={},
                 stopping_pobj=None, verbose=10, random_state=None):
        super().__init__(
            n_atoms, n_times_atom, reg=reg, n_iter=n_iter,
            solver_z=solver_z, solver_z_kwargs=solver_z_kwargs,
            solver_d=solver_d, solver_d_kwargs=solver_d_kwargs,
            rank1=rank1, uv_constraint=uv_constraint,
            eps=1e-10, D_init=None, D_init_params={},
            algorithm='batch', lmbd_max='fixed', raise_on_increase=True,
            loss='l2', stopping_pobj=None, use_sparse_z=False, n_jobs=n_jobs,
            verbose=10, callback=None, random_state=None, name="BatchCDL")


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
                 rank1=True, uv_constraint='separate',
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
            lmbd_max='fixed', raise_on_increase=True, loss='l2', verbose=10,
            callback=None, stopping_pobj=None, use_sparse_z=False,
            name="OnlineCDL")
