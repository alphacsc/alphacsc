import numpy as np
from .loss_and_gradient import compute_X_and_objective_multi
from .update_z_multi import update_z_multi
from .utils import lil

DEFAULT_TOL_Z = 1e-3

# XXX check consistency / proper use!


def get_z_encoder_for(
        X,
        D_hat,
        n_atoms,
        atom_support,
        n_jobs,
        solver='l-bfgs',
        solver_kwargs=dict(),
        algorithm='batch',
        reg=0.1,
        loss='l2',
        loss_params=None,
        uv_constraint='auto',
        feasible_evaluation=False,
        use_sparse_z=False):
    """
    Returns a z encoder for the required solver.

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data on which to perform CSC.
    D_hat : array, shape (n_atoms, n_channels, n_times) or
        (n_atoms, n_channels + atom_support)
        The dictionary used to encode the signal X. Can be either in the form
        of a full rank dictionary D (n_atoms, n_channels, atom_support) or with
        the spatial and temporal atoms uv (n_atoms, n_channels + atom_support)
    n_atoms : int
        The number of atoms to learn.
    atom_support : int
        The support of the atom.
    n_jobs : int
        The number of parallel jobs.
    solver : str
        The solver to use for the z update. Options are
        {{'l_bfgs' (default) | 'lgcd' | 'dicodile'}}.
    solver_kwargs : dict
        Additional keyword arguments to pass to update_z_multi.
    algorithm : 'batch' (default) | 'greedy' | 'online' | 'stochastic'
        Dictionary learning algorithm.
    reg : float
        The regularization parameter.
    loss : {{ 'l2' (default) | 'dtw' | 'whitening'}}
        Loss for the data-fit term. Either the norm l2 or the soft-DTW.
        If solver is 'dicodile', then the loss must be 'l2'.
    loss_params : dict | None
        Parameters of the loss.
        If solver_z is 'dicodile', then loss_params should be None.
    uv_constraint : {{'auto' | 'joint' | 'separate'}}
        The kind of norm constraint on the atoms:

        - :code:`'joint'`: the constraint is ||[u, v]||_2 <= 1
        - :code:`'separate'`: the constraint is ||u||_2 <= 1 and ||v||_2 <= 1

        If solver_z is 'dicodile', then uv_constraint must be auto.
    feasible_evaluation : boolean, default False
        If feasible_evaluation is True, it first projects on the feasible set,
        i.e. norm(uv_hat) <= 1.
        If solver_z is 'dicodile', then feasible_evaluation must be False.
    use_sparse_z : bool, default False
        Use sparse lil_matrices to store the activations.
        If solver_z is 'dicodile', then use_sparse_z must be False.

    Returns
    -------
    enc : instance of ZEncoder
        The encoder.
    """
    assert isinstance(solver_kwargs, dict), (
        'solver_kwargs should be a valid dictionary.'
    )

    assert (X is not None and len(X.shape) == 3), (
        'X should be a valid array of shape (n_trials, n_channels, n_times).'
    )

    assert (D_hat is not None and len(D_hat.shape) in [2, 3]), (
        'D_hat should be a valid array of shape '
        '(n_atoms, n_channels, n_times) '
        'or (n_atoms, n_channels + atom_support).'
    )

    assert algorithm in ['batch', 'greedy', 'online', 'stochastic'], (
        f'unrecognized algorithm type: {algorithm}.'
    )

    assert reg is not None, 'reg value cannot be None.'

    assert loss in ['l2', 'dtw', 'whitening'], (
        f'unrecognized loss type: {loss}.'
    )

    assert (loss_params is None) or isinstance(loss_params, dict), (
        'loss_params should be a valid dict or None.'
    )

    assert uv_constraint in ['joint', 'separate', 'auto'], (
        f'unrecognized uv_constraint type: {uv_constraint}.'
    )

    if solver in ['l-bfgs', 'lgcd']:
        if uv_constraint == 'auto':
            uv_constraint = 'separate'

        return AlphaCSCEncoder(
            X,
            D_hat,
            n_atoms,
            atom_support,
            n_jobs,
            solver,
            solver_kwargs,
            algorithm,
            reg,
            loss,
            loss_params,
            uv_constraint,
            feasible_evaluation,
            use_sparse_z)
    elif solver == 'dicodile':
        assert loss == 'l2', f"DiCoDiLe requires a l2 loss ('{loss}' passed)."
        assert loss_params is None, "DiCoDiLe requires loss_params=None."
        assert feasible_evaluation is False, (
            "DiCoDiLe requires feasible_evaluation=False."
        )
        assert uv_constraint == 'auto',  (
            "DiCoDiLe requires uv_constraint=auto."
        )
        assert use_sparse_z is False, (
            "DiCoDiLe requires use_sparse_z=False."
        )

        return DicodileEncoder(
            X,
            D_hat,
            n_atoms,
            atom_support,
            n_jobs,
            solver_kwargs,
            algorithm,
            reg,
            loss
        )
    else:
        raise ValueError(f'unrecognized solver type: {solver}.')


class BaseZEncoder:

    def __init__(
            self,
            X,
            D_hat,
            n_atoms,
            atom_support,
            n_jobs,
            solver_kwargs,
            algorithm,
            reg,
            loss):

        self.X = X
        self.D_hat = D_hat
        self.n_atoms = n_atoms
        self.atom_support = atom_support
        self.n_jobs = n_jobs

        self.solver_kwargs = solver_kwargs
        self.algorithm = algorithm
        self.reg = reg
        self.loss = loss

        self.n_trials, self.n_channels, self.n_times = X.shape
        self.n_times_valid = self.n_times - self.atom_support + 1

        self.constants = {}
        self.constants['n_channels'] = self.n_channels
        self.constants['XtX'] = np.dot(X.ravel(), X.ravel())

    def compute_z(self):
        """
        Perform one incremental z update.
        This is the "main" function of the algorithm.
        """
        raise NotImplementedError()

    def compute_z_partial(self, i0):
        """
        Compute z on a slice of the signal X, for online learning.

        Parameters
        ----------
        i0 : int
            Slice index.
        """
        raise NotImplementedError()

    def get_cost(self):
        """
        Computes the cost of the current sparse representation (z_hat)

        Returns
        -------
        cost: float
        """
        raise NotImplementedError()

    def get_sufficient_statistics(self):
        """
        Computes sufficient statistics to update D.

        Returns
        -------
        ztz, ztX : (ndarray, ndarray)
            Sufficient statistics.
        """
        raise NotImplementedError()

    def get_sufficient_statistics_partial(self):
        """
        Returns the partial sufficient statistics
        that were computed during the last call to
        compute_z_partial.

        Returns
        -------
        ztz, ztX : (ndarray, ndarray)
            Sufficient statistics for the slice that was
            selected in the last call of ``compute_z_partial``
        """
        raise NotImplementedError()

    def get_z_hat(self):
        """
        Returns a sparse encoding of the signal.

        Returns
        -------
        z_hat
            Sparse encoding of the signal X.
        """
        raise NotImplementedError()

    def set_D(self, D):
        """
        Update the dictionary.

        Parameters
        ----------
        D : ndarray, shape (n_atoms, n_channels, n_time_atoms)
            An updated dictionary, to be used for the next
            computation of z_hat.
        """
        raise NotImplementedError()

    def set_reg(self, reg):
        """
        Update the regularization parameter.

        Parameters
        ----------
        reg : float
              Regularization parameter
        """
        raise NotImplementedError()

    def get_constants(self):
        """
        """

        return self.constants

    def add_one_atom(self, new_atom):
        """
        Add one atom to the dictionary and extend z_hat
        to match the new dimensions.

        Parameters
        ----------
        new_atom : array, shape (n_channels + n_times_atom)
            A new atom to add to the dictionary.
        """
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class AlphaCSCEncoder(BaseZEncoder):
    def __init__(
            self,
            X,
            D_hat,
            n_atoms,
            atom_support,
            n_jobs,
            solver,
            solver_kwargs,
            algorithm,
            reg,
            loss,
            loss_params,
            uv_constraint,
            feasible_evaluation,
            use_sparse_z):

        super().__init__(X,
                         D_hat,
                         n_atoms,
                         atom_support,
                         n_jobs,
                         solver_kwargs,
                         algorithm,
                         reg,
                         loss)

        if loss_params is None:
            loss_params = dict(gamma=.1, sakoe_chiba_band=10, ordar=10)

        self.solver = solver
        self.loss_params = loss_params
        self.uv_constraint = uv_constraint
        self.feasible_evaluation = feasible_evaluation
        self.use_sparse_z = use_sparse_z

        self._init_z_hat()

    def _init_z_hat(self):

        self.z_hat = lil.init_zeros(
            self.use_sparse_z, self.n_trials, self.n_atoms, self.n_times_valid)

        if self.algorithm == 'greedy':
            # remove all atoms
            self.D_hat = self.D_hat[0:0]
            # remove all activations
            use_sparse_z = lil.is_list_of_lil(self.z_hat)
            n_trials, _, n_times_valid = lil.get_z_shape(self.z_hat)
            self.z_hat = lil.init_zeros(
                use_sparse_z, n_trials, 0, n_times_valid)

    def _compute_z_aux(self, X, z0, unbiased_z_hat):
        reg = self.reg if not unbiased_z_hat else 0

        return update_z_multi(
            X,
            self.D_hat,
            reg=reg,
            z0=z0,
            solver=self.solver,
            solver_kwargs=self.solver_kwargs,
            freeze_support=unbiased_z_hat,
            loss=self.loss,
            loss_params=self.loss_params,
            n_jobs=self.n_jobs,
            return_ztz=True)

    def compute_z(self, unbiased_z_hat=False):
        self.z_hat, self.constants['ztz'], self.constants['ztX'] = self._compute_z_aux(  # noqa
            self.X, self.z_hat, unbiased_z_hat)

    def compute_z_partial(self, i0, alpha=.8):
        if 'ztz' not in self.constants:
            self.constants['ztz'] = np.zeros(
                (self.n_atoms, self.n_atoms, 2 * self.atom_support - 1))
        if 'ztX' not in self.constants:
            self.constants['ztX'] = np.zeros(
                (self.n_atoms, self.n_channels, self.atom_support))

        self.z_hat[i0], self.ztz_i0, self.ztX_i0 = self._compute_z_aux(
            self.X[i0], self.z_hat[i0], unbiased_z_hat=False)

        self.constants['ztz'] = alpha * self.constants['ztz'] + self.ztz_i0
        self.constants['ztX'] = alpha * self.constants['ztX'] + self.ztX_i0

    def get_cost(self):
        cost = compute_X_and_objective_multi(self.X,
                                             self.z_hat,
                                             self.D_hat,
                                             reg=self.reg,
                                             loss=self.loss,
                                             loss_params=self.loss_params,
                                             uv_constraint=self.uv_constraint,
                                             feasible_evaluation=True,
                                             return_X_hat=False)
        return cost

    def get_sufficient_statistics(self):
        assert 'ztz' in self.constants and 'ztX' in self.constants, (
            'compute_z should be called to access the statistics.'
        )
        return self.constants['ztz'], self.constants['ztX']

    def get_sufficient_statistics_partial(self):
        assert hasattr(self, 'ztz_i0') and hasattr(self, 'ztX_i0'), (
            'compute_z_partial should be called to access the statistics.'
        )
        return self.ztz_i0, self.ztX_i0

    def set_D(self, D):
        self.D_hat = D

    def set_reg(self, reg):
        self.reg = reg

    def add_one_atom(self, new_atom):
        assert new_atom.shape == (self.atom_support + self.X.shape[1],)
        self.D_hat = np.concatenate([self.D_hat, new_atom[None]])
        self.z_hat = lil.add_one_atom_in_z(self.z_hat)

    def get_z_hat(self):
        return self.z_hat


class DicodileEncoder(BaseZEncoder):
    def __init__(
            self,
            X,
            D_hat,
            n_atoms,
            atom_support,
            n_jobs,
            solver_kwargs,
            algorithm,
            reg,
            loss):
        try:
            import dicodile
        except ImportError as ie:
            raise ImportError(
                'Please install DiCoDiLe by running '
                '"pip install alphacsc[dicodile]"') from ie

        super().__init__(X,
                         D_hat,
                         n_atoms,
                         atom_support,
                         n_jobs,
                         solver_kwargs,
                         algorithm,
                         reg,
                         loss)

        self._encoder = dicodile.update_z.distributed_sparse_encoder.DistributedSparseEncoder(  # noqa: E501
            n_workers=n_jobs)

        # DiCoDiLe only supports learning from one signal at a time,
        # and expect a signal of shape (n_channels, *sig_support)
        # whereas AlphaCSC requires a signal of
        # shape (n_trials, n_channels, n_times)
        assert X.shape[0] == 1, (
            "X should be a valid array of shape (1, n_channels, n_times)."
        )

        n_times = X.shape[2]
        self.X = X[0]
        self.D_hat = D_hat
        self.n_times_valid = n_times - atom_support + 1
        self.n_atoms = n_atoms
        self.atom_support = atom_support
        self.algorithm = algorithm

        tol = DEFAULT_TOL_Z * np.std(self.X)

        params = dicodile._dicodile.DEFAULT_DICOD_KWARGS.copy()
        # DiCoDiLe defaults
        # Impose z_positive = True, as in alphacsc z is always considered
        # positive
        params.update(tol=tol, reg=reg, timing=False,
                      z_positive=True, return_ztz=False, warm_start=True,
                      freeze_support=False, random_state=None)
        params.update(solver_kwargs)
        self.params = params
        self._encoder.init_workers(self.X, self.D_hat, reg, self.params)

    def compute_z(self):
        """
        Perform one incremental z update.
        This is the "main" function of the algorithm.
        """
        self.run_statistics = self._encoder.process_z_hat()

    def compute_z_partial(self, i0):
        """
        Compute z on a slice of the signal X, for online learning.

        Parameters
        ----------
        i0 : int
            Slice index.
        """
        raise NotImplementedError(
            "compute_z_partial is not available in DiCoDiLe")

    def get_cost(self):
        """
        Computes the cost of the current sparse representation (z_hat)

        Returns
        -------
        cost: float
        """
        if hasattr(self, 'run_statistics'):
            return self._encoder.get_cost()

        # If compute_z has not been run, return the value of cost function when
        # z_hat = 0
        return 0.5 * np.linalg.norm(self.X) ** 2

    def get_sufficient_statistics(self):
        """
        Computes sufficient statistics to update D.

        Returns
        -------
        ztz, ztX : (ndarray, ndarray)
            Sufficient statistics.
        """
        assert hasattr(self, 'run_statistics'), (
            'compute_z should be called to access the statistics.'
        )
        return self._encoder.get_sufficient_statistics()

    def get_sufficient_statistics_partial(self):
        """
        Returns the partial sufficient statistics
        that were computed during the last call to
        compute_z_partial.

        Returns
        -------
        ztz, ztX : (ndarray, ndarray)
            Sufficient statistics for the slice that was
            selected in the last call of ``compute_z_partial``
        """
        raise NotImplementedError(
            "Partial sufficient statistics are not available in DiCoDiLe")

    def get_z_hat(self):
        """
        Returns a sparse encoding of the signal.

        Returns
        -------
        z_hat : shape (n_trials, n_atoms, n_times_valid)
            Sparse encoding of the signal X.
        """
        if hasattr(self, 'run_statistics'):
            return self._encoder.get_z_hat()[None]

        # If compute_z has not been run, return 0.
        return np.zeros([1, self.n_atoms, self.n_times_valid])

    def set_D(self, D):
        """
        Update the dictionary.

        Parameters
        ----------
        D : ndarray, shape (n_atoms, n_channels, n_time_atoms)
            An updated dictionary, to be used for the next
            computation of z_hat.
        """
        self.D_hat = D
        self._encoder.set_worker_D(D)

    def set_reg(self, reg):
        """
        Update the regularization parameter.

        Parameters
        ----------
        reg : float
              Regularization parameter
        """
        self._encoder.set_worker_params({'reg': reg})  # XXX

    def add_one_atom(self, new_atom):
        """
        Add one atom to the dictionary and extend z_hat
        to match the new dimensions.

        Parameters
        ----------
        new_atom : array, shape (n_channels + n_times_atom)
            A new atom to add to the dictionary.
        """
        raise NotImplementedError(
            "Greedy learning is not available in DiCoDiLe")

    def __enter__(self):
        # XXX run init here?
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._encoder.release_workers()
        self._encoder.shutdown_workers()
