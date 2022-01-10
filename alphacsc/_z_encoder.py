import numpy as np

from .utils import construct_X_multi
from .utils.dictionary import get_D_shape
from .update_z_multi import update_z_multi
from .utils.dictionary import _patch_reconstruction_error
from .loss_and_gradient import compute_objective

DEFAULT_TOL_Z = 1e-3

# XXX check consistency / proper use!


def get_z_encoder_for(X, D_hat, n_atoms, n_times_atom, n_jobs,
                      solver='l-bfgs', solver_kwargs=dict(),
                      reg=0.1, loss='l2', loss_params=None):
    """
    Returns a z encoder for the required solver.

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data on which to perform CSC.
    D_hat : array, shape (n_atoms, n_channels, n_times) or
        (n_atoms, n_channels + n_times_atom)
        The dictionary used to encode the signal X. Can be either in the form
        of a full rank dictionary D (n_atoms, n_channels, n_times_atom) or with
        the spatial and temporal atoms uv (n_atoms, n_channels + n_times_atom)
    n_atoms : int
        The number of atoms to learn.
    n_times_atom : int
        The support of the atom.
    n_jobs : int
        The number of parallel jobs.
    solver : str
        The solver to use for the z update. Options are
        {{'l_bfgs' (default) | 'lgcd' | 'dicodile'}}.
    solver_kwargs : dict
        Additional keyword arguments to pass to update_z_multi.
    reg : float
        The regularization parameter.
    loss : {{ 'l2' (default) | 'dtw' | 'whitening'}}
        Loss for the data-fit term. Either the norm l2 or the soft-DTW.
        If solver is 'dicodile', then the loss must be 'l2'.
    loss_params : dict | None
        Parameters of the loss.
        If solver_z is 'dicodile', then loss_params should be None.

    Returns
    -------
    enc : instance of ZEncoder
        The encoder.
    """
    assert isinstance(solver_kwargs, dict), (
        'solver_kwargs should be a valid dictionary.'
    )

    assert (X is not None and X.ndim == 3), (
        'X should be a valid array of shape (n_trials, n_channels, n_times).'
    )

    assert (D_hat is not None and D_hat.ndim in [2, 3]), (
        'D_hat should be a valid array of shape '
        '(n_atoms, n_channels, n_times) '
        'or (n_atoms, n_channels + n_times_atom).'
    )

    assert reg is not None, 'reg value cannot be None.'

    assert loss in ['l2', 'dtw', 'whitening'], (
        f'unrecognized loss type: {loss}.'
    )

    assert (loss_params is None) or isinstance(loss_params, dict), (
        'loss_params should be a valid dict or None.'
    )

    if solver in ['l-bfgs', 'lgcd']:

        return AlphaCSCEncoder(
            X, D_hat, n_atoms, n_times_atom, n_jobs,
            solver, solver_kwargs, reg, loss, loss_params
        )

    elif solver == 'dicodile':
        assert loss == 'l2', f"DiCoDiLe requires a l2 loss ('{loss}' passed)."
        assert loss_params is None, "DiCoDiLe requires loss_params=None."

        return DicodileEncoder(
            X, D_hat, n_atoms, n_times_atom, n_jobs,
            solver_kwargs, reg, loss, loss_params
        )
    else:
        raise ValueError(f'unrecognized solver type: {solver}.')


class BaseZEncoder:

    def __init__(self, X, D_hat, n_atoms, n_times_atom, n_jobs,
                 solver_kwargs, reg, loss, loss_params):

        self.X = X
        self.D_hat = D_hat
        self.n_atoms = n_atoms
        self.n_times_atom = n_times_atom
        self.n_jobs = n_jobs

        self.solver_kwargs = solver_kwargs
        self.reg = reg
        self.loss = loss
        self.loss_params = loss_params

        self.n_trials, self.n_channels, self.n_times = X.shape
        self.n_times_valid = self.n_times - self.n_times_atom + 1

        self.XtX = np.dot(X.ravel(), X.ravel())

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

    def compute_objective(self, D):
        '''Compute the value of the objective function.

        Parameters
        ----------
        D : array, shape (n_atoms, n_channels + n_times_atom) or
                         (n_atoms, n_channels, n_times_atom)
            The atoms to learn from the data.
            D should be feasible.

        Returns
        -------
        obj :
            The value of objective function.
        '''
        if self.loss == 'l2':
            return compute_objective(D=D, constants=self.get_constants())

        else:
            X_hat = construct_X_multi(
                self.z_hat, D=D, n_channels=self.n_channels)

            return compute_objective(
                X=self.X, X_hat=X_hat, z_hat=self.z_hat,
                reg=self.reg, loss=self.loss,
                loss_params=self.loss_params
            )

    def get_cost(self):
        """
        Computes the cost of the current sparse representation (z_hat)

        Returns
        -------
        cost: float
            The value of the objective function
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
        Returns the partial sufficient statistics that were
        computed during the last call to compute_z_partial.

        Returns
        -------
        ztz, ztX : (ndarray, ndarray)
            Sufficient statistics for the slice that was
            selected in the last call of ``compute_z_partial``
        """
        raise NotImplementedError()

    def get_max_error_patch(self):
        """
        Returns the patch of the signal with the largest reconstuction error.

        Returns
        -------
        D_k : ndarray, shape (n_channels, n_times_atom) or
                (n_channels + n_times_atom,)
            Patch of the residual with the largest error.
        """
        raise NotImplementedError()

    def get_z_hat(self):
        """
        Returns the sparse codes of the signals.

        Returns
        -------
        z_hat : ndarray, shape (n_trials, n_atoms, n_times_valid)
            Sparse codes of the signal X.
        """
        raise NotImplementedError()

    def get_z_nnz(self):
        """
        Return the number of non-zero activations per atoms for the signals.

        Returns
        -------
        z_nnz : ndarray, shape (n_atoms,)
            Ratio of non-zero activations for each atom.
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

        return dict(n_channels=self.n_channels, XtX=self.XtX,
                    ztz=self.ztz, ztX=self.ztX)

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
    def __init__(self, X, D_hat, n_atoms, n_times_atom, n_jobs,
                 solver, solver_kwargs, reg, loss, loss_params):

        if loss_params is None:
            loss_params = dict(gamma=.1, sakoe_chiba_band=10, ordar=10)

        super().__init__(
            X, D_hat, n_atoms, n_times_atom, n_jobs,
            solver_kwargs, reg, loss, loss_params
        )

        self.solver = solver

        effective_n_atoms = self.D_hat.shape[0]
        self.z_hat = self._get_new_z_hat(effective_n_atoms)

    def _get_new_z_hat(self, n_atoms):
        """
        Returns a array filed with 0 with the right size for sparse codes.
        """
        return np.zeros((
            self.n_trials, n_atoms, self.n_times_valid
        ))

    def _compute_z_aux(self, X, z0, unbiased_z_hat):
        reg = self.reg if not unbiased_z_hat else 0

        return update_z_multi(
            X, self.D_hat, reg=reg, z0=z0, solver=self.solver,
            solver_kwargs=self.solver_kwargs, freeze_support=unbiased_z_hat,
            loss=self.loss, loss_params=self.loss_params, n_jobs=self.n_jobs,
            return_ztz=True
        )

    def compute_z(self, unbiased_z_hat=False):
        self.z_hat, self.ztz, self.ztX = self._compute_z_aux(self.X,
                                                             self.z_hat,
                                                             unbiased_z_hat)

    def compute_z_partial(self, i0, alpha=.8):
        if not hasattr(self, 'ztz'):
            self.ztz = np.zeros(
                (self.n_atoms, self.n_atoms, 2 * self.n_times_atom - 1))
        if not hasattr(self, 'ztX'):
            self.ztX = np.zeros(
                (self.n_atoms, self.n_channels, self.n_times_atom))

        self.z_hat[i0], self.ztz_i0, self.ztX_i0 = self._compute_z_aux(
            self.X[i0], self.z_hat[i0], unbiased_z_hat=False)

        self.ztz = alpha * self.ztz + self.ztz_i0
        self.ztX = alpha * self.ztX + self.ztX_i0

    def get_cost(self):

        X_hat = construct_X_multi(self.z_hat, D=self.D_hat,
                                  n_channels=self.n_channels)

        return compute_objective(X=self.X, X_hat=X_hat, z_hat=self.z_hat,
                                 reg=self.reg, loss=self.loss,
                                 loss_params=self.loss_params)

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

    def get_max_error_patch(self):
        """
        Returns the patch of the signal with the largest reconstuction error.

        Returns
        -------
        D_k : ndarray, shape (n_channels, n_times_atom) or
                (n_channels + n_times_atom,)
            Patch of the residual with the largest error.
        """
        patch_rec_error = _patch_reconstruction_error(
            self.X, self.z_hat, self.D_hat
        )
        i0 = patch_rec_error.argmax()
        n0, t0 = np.unravel_index(i0, patch_rec_error.shape)

        n_channels = self.X.shape[1]
        *_, n_times_atom = get_D_shape(self.D_hat, n_channels)
        return self.X[n0, :, t0:t0 + n_times_atom][None]

    def set_D(self, D):
        self.D_hat = D

    def set_reg(self, reg):
        self.reg = reg

    def add_one_atom(self, new_atom):
        assert new_atom.shape == (self.n_times_atom + self.X.shape[1],)
        self.D_hat = np.concatenate([self.D_hat, new_atom[None]])
        self.z_hat = np.concatenate(
            [self.z_hat, self._get_new_z_hat(1)], axis=1
        )

    def get_z_hat(self):
        return self.z_hat

    def get_z_nnz(self):
        """
        Return the number of non-zero activations per atoms for the signals.

        Returns
        -------
        z_nnz : ndarray, shape (n_atoms,)
            Ratio of non-zero activations for each atom.
        """
        z_nnz = np.sum(self.z_hat != 0, axis=(0, 2))
        z_size = self.z_hat.size / self.z_hat.shape[1]
        return z_nnz / z_size


class DicodileEncoder(BaseZEncoder):
    def __init__(self, X, D_hat, n_atoms, n_times_atom, n_jobs,
                 solver_kwargs, reg, loss, loss_params):
        try:
            import dicodile
        except ImportError as ie:
            raise ImportError(
                'Please install DiCoDiLe by running '
                '"pip install alphacsc[dicodile]"') from ie

        super().__init__(
            X, D_hat, n_atoms, n_times_atom, n_jobs,
            solver_kwargs, reg, loss, loss_params
        )

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
        self.D_hat = D_hat
        self.n_times_valid = n_times - n_times_atom + 1
        self.n_atoms = n_atoms
        self.n_times_atom = n_times_atom

        tol = DEFAULT_TOL_Z * np.std(self.X[0])

        params = dicodile._dicodile.DEFAULT_DICOD_KWARGS.copy()
        # DiCoDiLe defaults
        # Impose z_positive = True, as in alphacsc z is always considered
        # positive
        params.update(tol=tol, reg=reg, timing=False,
                      z_positive=True, return_ztz=False, warm_start=True,
                      freeze_support=False, random_state=None)
        params.update(solver_kwargs)
        self.params = params
        self._encoder.init_workers(self.X[0], self.D_hat, reg, self.params)

    def compute_z(self):
        """
        Perform one incremental z update.
        This is the "main" function of the algorithm.
        """
        self.run_statistics = self._encoder.process_z_hat()
        self.ztz, self.ztX = self._encoder.get_sufficient_statistics()

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
            The value of the objective function.
        """
        if hasattr(self, 'run_statistics'):
            return self._encoder.get_cost()

        # If compute_z has not been run, return the value of cost function when
        # z_hat = 0
        return 0.5 * np.linalg.norm(self.X[0]) ** 2

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
        return self.ztz, self.ztX

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
            "Partial sufficient statistics are not available in DiCoDiLe"
        )

    def get_max_error_patch(self):
        """
        Returns the patch of the signal with the largest reconstuction error.

        Returns
        -------
        D_k : ndarray, shape (n_channels, n_times_atom) or
                (n_channels + n_times_atom,)
            Patch of the residual with the largest error.
        """
        # XXX - this step should be implemented in dicodile
        # See issue tommoral/dicodile#49
        patch_rec_error = _patch_reconstruction_error(
            self.X, self.get_z_hat(), self.D_hat
        )
        i0 = patch_rec_error.argmax()
        n0, t0 = np.unravel_index(i0, patch_rec_error.shape)

        n_channels = self.X.shape[1]
        *_, n_times_atom = get_D_shape(self.D_hat, n_channels)
        return self.X[n0, :, t0:t0 + n_times_atom][None]

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

    def get_z_nnz(self):
        """
        Return the number of non-zero activations per atoms for the signals.

        Returns
        -------
        z_nnz : ndarray, shape (n_atoms,)
            Ratio of non-zero activations for each atom.
        """
        effective_n_atoms = self.D_hat.shape[0]
        if not hasattr(self, 'run_statistics'):
            return np.zeros(effective_n_atoms)

        z_nnz = self._encoder.get_z_nnz()
        return z_nnz / z_nnz.shape[-1]

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
