import numpy as np

from .init_dict import get_init_strategy
from .loss_and_gradient import compute_objective
from .utils.dictionary import get_lambda_max, NoWindow
from .utils.optim import fista
from .utils.validation import check_random_state


class BaseZEncoder:
    """Base class for a z encoder.

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data on which to perform CSC.
    D_hat : array, shape (n_atoms, n_channels, n_times) or
        (n_atoms, n_channels + n_times_atom)
        The dictionary used to encode the signal X. Can be either in the form
        of a full rank dictionary D (n_atoms, n_channels, n_times_atom) or with
        the spatial and temporal atoms uv (n_atoms, n_channels + n_times_atom)
    n_jobs : int
        The number of parallel jobs.
    solver_kwargs : dict
        Additional keyword arguments to pass to the encoder.
    reg : float
        The regularization parameter.
    """

    def __init__(self, X, D_hat, n_jobs, solver_kwargs, reg):

        self.X = X
        self.D_hat = D_hat
        self.n_jobs = n_jobs

        self.solver_kwargs = solver_kwargs
        self.reg = reg

        _, self.n_channels, self.n_times = X.shape
        self.n_times_atom = D_hat.shape[-1]
        if D_hat.ndim == 2:
            self.n_times_atom -= self.n_channels
        self.n_times_valid = self.n_times - self.n_times_atom + 1

        self.XtX = np.dot(X.ravel(), X.ravel())

    def compute_z(self):
        """
        Encode X by computing z for the current dictionary D_hat.

        This is the "main" method of this class.
        """
        raise NotImplementedError()

    def compute_z_partial(self, i0):
        """
        Encode a slice of the signal X, for online learning.

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
        return compute_objective(D=D, constants=self.get_constants())

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
        Returns z_hat, corresponding to the encoded signal X.

        Returns
        -------
        z_hat : ndarray, shape (n_trials, n_atoms, n_times_valid)
            Sparse codes of the signal X.
        """
        raise NotImplementedError()

    def get_z_hat_shape(self):
        """
        Returns the shape of the sparse codes.

        Returns
        -------
        shape : tuple
            Shape of the sparse code.
        """
        return (self.X.shape[0], self.D_hat.shape[0], self.n_times_valid)

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
        Update the encoder parameters, usually a dictionary.

        Parameters
        ----------
        D : ndarray, shape (n_atoms, n_channels, n_time_atoms)
            An updated dictionary, to be used for the next
            computation of z_hat.
        """
        raise NotImplementedError()

    def update_reg(self, is_per_atom):
        """
        Update the regularization parameter.

        Parameters
        ----------
        is_per_atom: bool
            True if lmbd_max='per_atom'; False otherwise

        """
        self.reg = self.reg * get_lambda_max(self.X, self.D_hat)

        if not is_per_atom:
            self.reg = self.reg.max()

    def get_constants(self):
        """
        Return information necessary to update the encoder parameter.

        For dictionary learning, this corresponds to the signal gram-matrix
        $X^TX$, and the sufficient statistics $Z^TZ, Z^T X$.
        """

        return dict(n_channels=self.n_channels, XtX=self.XtX,
                    ztz=self.ztz, ztX=self.ztX)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class BaseDSolver:
    """Base class for a d solver."""

    def __init__(self, n_channels, n_atoms, n_times_atom,
                 uv_constraint, D_init, init_kwargs,
                 resample_strategy, window, eps, max_iter,
                 momentum, random_state, verbose, debug):

        self.n_channels = n_channels
        self.n_atoms = n_atoms
        self.n_times_atom = n_times_atom
        self.uv_constraint = uv_constraint
        self.eps = eps
        self.max_iter = max_iter
        self.momentum = momentum
        self.rng = check_random_state(random_state)
        self.verbose = verbose
        self.debug = debug
        self.D_init = D_init

        self.init_strategy = get_init_strategy(
            n_times_atom, self.get_D_shape(), self.rng, D_init, init_kwargs
        )

        if not window:
            self._windower = NoWindow()
        else:
            self._init_windower()

        self.resample_strategy = resample_strategy
        assert self.resample_strategy in ['greedy', 'chunk', 'random'], (
            "resample_strategy should be greedy, chunk, or random. "
            f"Got resample_strategy='{self.resample_strategy}'."
        )

    def _get_objective(self, z_encoder):

        def objective(D, full=False):

            D = self._windower.window(D)

            return z_encoder.compute_objective(D)

        return objective

    def _get_prox(self):

        def prox(D, step_size=None):

            D = self._windower.window(D)

            D = self.prox(D)

            return self._windower.remove_window(D)

        return prox

    def _get_grad(self, z_encoder):

        def grad(D):

            D = self._windower.window(D)

            grad = self.grad(D, z_encoder)

            return self._windower.window(grad)

        return grad

    def init_dictionary(self, X):
        """Returns a dictionary for the signal X depending on D_init value.

        Parameters
        ----------
        X: array, shape (n_trials, n_channels, n_times)
            The data on which to perform CSC.

        Return
        ------
        D : array shape (n_atoms, n_channels + n_times_atom) or
                  shape (n_atoms, n_channels, n_times_atom)
            The initial atoms to learn from the data.
        """

        D_hat = self.init_strategy.initialize(X)

        if not isinstance(self.D_init, np.ndarray):
            D_hat = self._windower.window(D_hat)

        self.D_hat = self.prox(D_hat)

        return self.D_hat

    def get_max_error_subwindow(self, z_encoder):
        """Get the maximal reconstruction error sub-window from the data
        as a new atom.

        This idea is used for instance in [Yellin2017]

        Parameters
        ----------
        z_encoder : BaseZEncoder
            ZEncoder object to be able to compute the largest error sub-window.

        Return
        ------
        dk: array, shape (n_channels + n_times_atom,) or
                         (n_channels, n_times_atom,)
            New atom for the dictionary, chosen as the sub-windos of the signal
            with the maximal reconstruction error.

        [Yellin2017] BLOOD CELL DETECTION AND COUNTING IN HOLOGRAPHIC LENS-FREE
        IMAGING BY CONVOLUTIONAL SPARSE DICTIONARY LEARNING AND CODING.
        """
        assert z_encoder.n_channels == self.n_channels

        d0 = z_encoder.get_max_error_patch()

        d0 = self._windower.window(d0)

        return self.prox(d0)

    def add_one_atom(self, z_encoder):
        """Adds one atom to D_hat and updates D_hat in z_encoder.

        Parameters
        ----------
        z_encoder: BaseZEncoder
            ZEncoder object.

        Returns
        -------
        D_hat : array, shape (k+1, n_channels + n_times_atom) or
                             (k+1, n_channels, n_times_atom)
            The atoms to learn from the data, where k < n_atoms is the initial
        number of atoms in the dictionary before adding an atom.
        """
        assert self.D_hat.shape[0] < self.n_atoms

        new_atom = self.get_max_error_subwindow(z_encoder)[0]

        self.D_hat = np.concatenate([self.D_hat, new_atom[None]])

        z_encoder.set_D(self.D_hat)
        return self.D_hat

    def resample_atom(self, k0, z_encoder):
        """Resamples the atom at index k0 of D_hat and updates D_hat in
        z_encoder.

        Parameters
        ----------
        k0: int
            index of the atom to resample
        z_encoder: BaseZEncoder
            ZEncoder object.

        Returns
        -------
        D_hat : array, shape (n_atoms, n_channels + n_times_atom) or
                             (n_atoms, n_channels, n_times_atom)
            The atoms to learn from the data.
        """
        if self.resample_strategy == 'greedy':
            # TODO: This function is sometimes called with z_nnz[k0] = 1
            # In that case, the max_error_subwindow should be taken after
            # removing the atom k0 in the reconstruction
            self.D_hat[k0] = self.get_max_error_subwindow(z_encoder)[0]
        elif self.resample_strategy in ['chunk', 'random']:
            from .init_dict import init_dictionary as init_dict
            self.D_hat[k0] = init_dict(
                z_encoder.X, 1, self.n_times_atom, self.uv_constraint,
                rank1=self.rank1, window=(self._windower != NoWindow()),
                D_init=self.resample_strategy, random_state=None
            )
        else:
            raise NotImplementedError(
                f"Unknown resampling strategy '{self.resample_strategy}'"
            )

        z_encoder.set_D(self.D_hat)
        return self.D_hat

    def update_D(self, z_encoder):
        """Learn d's in time domain and update D_hat in z_encoder.

        Parameters
        ----------
        z_encoder: BaseZEncoder
            ZEncoder object.

        Returns
        -------
        D_hat : array, shape (n_atoms, n_channels + n_times_atom) or
                             (n_atoms, n_channels, n_times_atom)
            The atoms to learn from the data.
        """

        assert z_encoder.n_channels == self.n_channels

        D_hat0 = self._windower.remove_window(self.D_hat)

        D_hat, pobj = fista(
            self._get_objective(z_encoder), self._get_grad(z_encoder),
            self._get_prox(), None, D_hat0, self.max_iter,
            momentum=self.momentum, eps=self.eps, adaptive_step_size=True,
            name=self.name, debug=self.debug, verbose=self.verbose
        )

        self.D_hat = self._windower.window(D_hat)

        z_encoder.set_D(self.D_hat)

        if self.debug:
            return self.D_hat, pobj
        return self.D_hat

    def get_D_shape(self):
        """Returns the expected shape of the dictionary.

        Note: For the 'greedy' strategy this does not return the actual
        dictionary shape, but the final expected shape.
        """
        return (self.n_atoms, self.n_channels, self.n_times_atom)
