import numpy as np
from .loss_and_gradient import compute_X_and_objective_multi
from .update_z_multi import update_z_multi
from .utils import check_dimension, lil


# XXX check consistency / proper use!
def get_z_encoder_for(
        X,
        D_hat,
        n_atoms,
        atom_support,
        n_jobs,
        solver='l-bfgs',
        z_kwargs=dict(),
        algorithm='batch',
        reg=0.1,
        loss='l2',
        loss_params=None,
        uv_constraint='separate',
        feasible_evaluation=True,
        use_sparse_z=False):
    """
    Returns a z encoder for the required solver.

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data on which to perform CSC.
    D_hat : array, shape (n_trials, n_channels, n_times) or
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
        {{'l_bfgs' (default) | 'lgcd'}}.
    z_kwargs : dict
        Additional keyword arguments to pass to update_z_multi.
    algorithm : 'batch' (default) | 'greedy' | 'online' | 'stochastic'
        Dictionary learning algorithm.
    reg : float
        The regularization parameter.
    loss : {{ 'l2' (default) | 'dtw' | 'whitening'}}
        Loss for the data-fit term. Either the norm l2 or the soft-DTW.
    loss_params : dict | None
        Parameters of the loss.
    uv_constraint : {{'joint' | 'separate'}}
        The kind of norm constraint on the atoms:

        - :code:`'joint'`: the constraint is ||[u, v]||_2 <= 1
        - :code:`'separate'`: the constraint is ||u||_2 <= 1 and ||v||_2 <= 1
    feasible_evaluation : boolean, default True
        If feasible_evaluation is True, it first projects on the feasible set,
        i.e. norm(uv_hat) <= 1.
    use_sparse_z : bool, default False
        Use sparse lil_matrices to store the activations.

    Returns
    -------
    enc : instance of ZEncoder
        The encoder.
    """
    assert isinstance(z_kwargs, dict), 'z_kwargs should be a valid dictionary.'

    assert (X is not None and len(X.shape) == 3), (
        'X should be a valid array of shape (n_trials, n_channels, n_times).'
    )

    assert (D_hat is not None and len(D_hat.shape) in [2, 3]), (
        'D_hat should be a valid array of shape '
        '(n_trials, n_channels, n_times) '
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

    assert uv_constraint in ['joint', 'separate'], (
        f'unrecognized uv_constraint type: {uv_constraint}.'
    )

    if solver in ['l-bfgs', 'lgcd']:
        return AlphaCSCEncoder(
            X,
            D_hat,
            n_atoms,
            atom_support,
            n_jobs,
            solver,
            z_kwargs,
            algorithm,
            reg,
            loss,
            loss_params,
            uv_constraint,
            feasible_evaluation,
            use_sparse_z)
    elif solver == 'dicodile':
        return DicodileEncoder(
            X,
            D_hat,
            n_atoms,
            atom_support,
            n_jobs,
            z_kwargs,
            algorithm,
            reg,
            loss,
            loss_params
        )
    else:
        raise ValueError(f'unrecognized solver type: {solver}.')


class BaseZEncoder:
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
            z_kwargs,
            algorithm,
            reg,
            loss,
            loss_params,
            uv_constraint,
            feasible_evaluation,
            use_sparse_z):

        if loss_params is None:
            loss_params = dict(gamma=.1, sakoe_chiba_band=10, ordar=10)

        self.X = X
        self.D_hat = D_hat
        self.n_atoms = n_atoms
        self.atom_support = atom_support
        self.n_jobs = n_jobs
        self.z_alg = solver
        self.z_kwargs = z_kwargs
        self.algorithm = algorithm
        self.reg = reg
        self.loss = loss
        self.loss_params = loss_params
        self.uv_constraint = uv_constraint
        self.feasible_evaluation = feasible_evaluation
        self.use_sparse_z = use_sparse_z

        self._init_z_hat()

    def _init_z_hat(self):
        n_trials, _, n_times = check_dimension(self.X)
        n_times_valid = n_times - self.atom_support + 1

        self.z_hat = lil.init_zeros(
            self.use_sparse_z, n_trials, self.n_atoms, n_times_valid)

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
            solver=self.z_alg,
            solver_kwargs=self.z_kwargs,
            freeze_support=unbiased_z_hat,
            loss=self.loss,
            loss_params=self.loss_params,
            n_jobs=self.n_jobs,
            return_ztz=True)

    def compute_z(self, unbiased_z_hat=False):
        self.z_hat, self.ztz, self.ztX = self._compute_z_aux(
            self.X, self.z_hat, unbiased_z_hat)

    def compute_z_partial(self, i0):
        self.z_hat[i0], self.ztz_i0, self.ztX_i0 = self._compute_z_aux(
            self.X[i0], self.z_hat[i0], unbiased_z_hat=False)

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
        assert hasattr(self, 'ztz') and hasattr(self, 'ztX'), (
            'compute_z should be called to access the statistics.'
        )
        return self.ztz, self.ztX

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
            z_kwargs,
            algorithm,
            reg,
            loss,
            loss_params):
        try:
            import dicodile

            self._encoder = dicodile.DistributedSparseEncoder(
               n_workers=n_jobs
            )

            self._encoder.init_workers(X, D_hat, reg, {})  # XXX params

            self.n_atoms = n_atoms
            self.atom_support = atom_support
            self.z_kwargs = z_kwargs
            self.algorithm = algorithm
            self.loss = loss
            self.loss_params = loss_params

        except ImportError as ie:
            raise ImportError(
                'Please install DiCoDiLe by running '
                '"pip install dicodile"') from ie

    def compute_z(self):
        """
        Perform one incremental z update.
        This is the "main" function of the algorithm.
        """
        self._encoder.process_z_hat()

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
        return self._encoder.get_cost()

    def get_sufficient_statistics(self):
        """
        Computes sufficient statistics to update D.

        Returns
        -------
        ztz, ztX : (ndarray, ndarray)
            Sufficient statistics.
        """
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
        z_hat
            Sparse encoding of the signal X.
        """
        return self._encoder.get_z_hat()

    def set_D(self, D):
        """
        Update the dictionary.

        Parameters
        ----------
        D : ndarray, shape (n_atoms, n_channels, n_time_atoms)
            An updated dictionary, to be used for the next
            computation of z_hat.
        """
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
