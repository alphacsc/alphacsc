import numpy as np

from .init_dict import get_init_strategy
from .loss_and_gradient import gradient_uv, gradient_d
from .utils.validation import check_random_state
from .utils.convolution import numpy_convolve_uv
from .utils.dictionary import NoWindow, UVWindower, SimpleWindower
from .utils.optim import fista, power_iteration
from .update_d_multi import prox_uv, prox_d


def check_solver_and_constraints(rank1, solver_d, uv_constraint):
    """Checks if solver_d and uv_constraint are compatible depending on
    rank1 value.

    - If rank1 is False, solver_d should be 'fista' and uv_constraint should be
    'auto'.
    - If rank1 is True;
       - If solver_d is either 'alternate' or 'alternate_adaptive',
         uv_constraint should be 'separate'.
       - If solver_d is either 'joint' or 'fista', uv_constraint should be
         'joint'.

    Parameters
    ----------
    rank1: boolean
        If set to True, learn rank 1 dictionary atoms.
    solver_d : str in {'alternate' | 'alternate_adaptive' | 'fista' | 'joint' |
    'auto'}
        The solver to use for the d update.
        - If rank1 is False, only option is 'fista'
        - If rank1 is True, options are 'alternate', 'alternate_adaptive'
          (default) or 'joint'
    uv_constraint : str in {'joint' | 'separate' | 'auto'}
        The kind of norm constraint on the atoms if using rank1=True.
        If 'joint', the constraint is norm_2([u, v]) <= 1
        If 'separate', the constraint is norm_2(u) <= 1 and norm_2(v) <= 1
        If rank1 is False, then uv_constraint must be 'auto'.
    """

    if rank1:
        if solver_d == 'auto':
            solver_d = 'alternate_adaptive'
        if 'alternate' in solver_d:
            if uv_constraint == 'auto':
                uv_constraint = 'separate'
            else:
                assert uv_constraint == 'separate', (
                    "solver_d='alternate*' should be used with "
                    f"uv_constraint='separate'. Got '{uv_constraint}'."
                )
        elif uv_constraint == 'auto' and solver_d in ['joint', 'fista']:
            uv_constraint = 'joint'
    else:
        assert solver_d in ['auto', 'fista'], (
            f"solver_d should be auto or fista. Got solver_d='{solver_d}'."
        )
        assert solver_d in ['auto', 'fista'] and uv_constraint == 'auto', (
            "If rank1 is False, uv_constraint should be 'auto' "
            f"and solver_d should be auto or fista. Got solver_d='{solver_d}' "
            f"and uv_constraint='{uv_constraint}'."
        )
        solver_d = 'fista'
    return solver_d, uv_constraint


def get_solver_d(n_channels, n_atoms, n_times_atom,
                 solver_d='alternate_adaptive', rank1=False,
                 uv_constraint='auto', D_init=None, resample_strategy='greedy',
                 window=False, eps=1e-8, max_iter=300, momentum=False,
                 random_state=None, verbose=0, debug=False):
    """Returns solver depending on solver_d type and rank1 value.

    Parameters
    ----------
    n_channels : int
        The number of channels.
    n_atoms : int
        The number of atoms to learn.
    n_times_atom : int
        The support of the atom.
    solver_d : str in {'alternate' | 'alternate_adaptive' | 'fista' | 'joint' |
    'auto'}
        The solver to use for the d update.
        - If rank1 is False, only option is 'fista'
        - If rank1 is True, options are 'alternate', 'alternate_adaptive'
          (default), 'joint' or 'fista'
    rank1: boolean
        If set to True, learn rank 1 dictionary atoms.
    uv_constraint : str in {'joint' | 'separate' | 'auto'}
        The kind of norm constraint on the atoms if using rank1=True.
        If 'joint', the constraint is norm_2([u, v]) <= 1
        If 'separate', the constraint is norm_2(u) <= 1 and norm_2(v) <= 1
        If rank1 is False, then uv_constraint must be 'auto'.
    D_init : {'chunk' | 'random' | 'greedy'}
             or array, shape (n_atoms, n_channels + n_times_atom) or
                         (n_atoms, n_channels, n_times_atom)
        The initialization scheme for the dictionary or the initial atoms.
    resample_strategy : {'greedy' | 'chunk' | 'random'}
        Resampling strategy to reset unused atoms. Can be either:
            - 'greedy': takes the signal sub-window with the largest
                        reconstruction error.
            - 'chunk': get a random sub-window from the original signal.
            - 'random': sample a random normal sub-window.
    window : boolean
        If True, re-parametrizes the atoms with a temporal Tukey window
    eps : float
        Stopping criterion. If the cost descent after a uv and a z update is
        smaller than eps, return.
    max_iter: int
        Number of iterations of gradient descent.
    momentum : bool
        If True, use an accelerated version of the proximal gradient descent.
    random_state : int | None
        The random state.
    verbose: int
        Verbosity level.
    debug : bool
        If True, return the cost at each iteration.

    """

    solver_d, uv_constraint = check_solver_and_constraints(rank1, solver_d,
                                                           uv_constraint)

    if rank1:
        if solver_d in ['auto', 'alternate', 'alternate_adaptive']:
            return AlternateDSolver(
                n_channels, n_atoms, n_times_atom, solver_d, uv_constraint,
                D_init, resample_strategy, window, eps, max_iter, momentum,
                random_state, verbose, debug
            )
        elif solver_d in ['fista', 'joint']:
            return JointDSolver(
                n_channels, n_atoms, n_times_atom, solver_d, uv_constraint,
                D_init, resample_strategy, window, eps, max_iter, momentum,
                random_state, verbose, debug
            )
        else:
            raise ValueError('Unknown solver_d: %s' % (solver_d, ))
    else:
        if solver_d in ['auto', 'fista']:
            return DSolver(
                n_channels, n_atoms, n_times_atom, solver_d, uv_constraint,
                D_init, resample_strategy, window, eps, max_iter, momentum,
                random_state, verbose, debug
            )
        else:
            raise ValueError('Unknown solver_d: %s' % (solver_d, ))


class BaseDSolver:
    """Base class for a d solver."""

    def __init__(self, n_channels, n_atoms, n_times_atom, solver_d,
                 uv_constraint, D_init, resample_strategy, window, eps,
                 max_iter, momentum, random_state, verbose, debug):

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
            n_times_atom, self.get_D_shape(), self.rng, D_init
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


class Rank1DSolver(BaseDSolver):
    """Base class for a rank1 solver d."""

    def __init__(self, n_channels, n_atoms, n_times_atom, solver_d,
                 uv_constraint, D_init, resample_strategy, window, eps,
                 max_iter, momentum, random_state, verbose, debug):

        super().__init__(
            n_channels, n_atoms, n_times_atom, solver_d, uv_constraint,
            D_init, resample_strategy, window, eps, max_iter, momentum,
            random_state, verbose, debug
        )

        self.name = "Update uv"
        self.rank1 = True

    def _init_windower(self):
        self._windower = UVWindower(self.n_times_atom, self.n_channels)

    def prox(self, D_hat):
        return prox_uv(D_hat, uv_constraint=self.uv_constraint,
                       n_channels=self.n_channels)

    def get_D_shape(self):
        """Returns the expected shape of the dictionary.

        Note: For the 'greedy' strategy this does not return the actual
        dictionary shape, but the final expected shape.
        """
        return (self.n_atoms, self.n_channels + self.n_times_atom)


class JointDSolver(Rank1DSolver):
    """A class for 'fista' or 'joint' solver_d when rank1 is True. """

    def __init__(self, n_channels, n_atoms, n_times_atom, solver_d,
                 uv_constraint, D_init, resample_strategy, window,
                 eps, max_iter, momentum, random_state, verbose, debug):

        super().__init__(
            n_channels, n_atoms, n_times_atom, solver_d, uv_constraint, D_init,
            resample_strategy,  window, eps, max_iter, momentum,
            random_state, verbose, debug
        )

    def grad(self, D, z_encoder):
        return gradient_uv(
            uv=D, X=z_encoder.X, z=None,
            constants=z_encoder.get_constants()
        )


class AlternateDSolver(Rank1DSolver):
    """A class for 'alternate' or 'alternate_adaptive' solver_d when rank1 is
       True.
    """

    def __init__(self, n_channels, n_atoms, n_times_atom, solver_d,
                 uv_constraint, D_init, resample_strategy, window, eps,
                 max_iter, momentum, random_state, verbose, debug):

        super().__init__(
            n_channels, n_atoms, n_times_atom, solver_d, uv_constraint, D_init,
            resample_strategy, window, eps, max_iter, momentum, random_state,
            verbose, debug
        )

        self.adaptive_step_size = (solver_d == 'alternate_adaptive')

    def update_D(self, z_encoder):
        """Learn d's in time domain.

        Parameters
        ----------
        z_encoder: BaseZEncoder
            ZEncoder object.

        Returns
        -------
        uv_hat : array, shape (n_atoms, n_channels + n_times_atom)
            The atoms to learn from the data.
        """
        assert z_encoder.n_channels == self.n_channels

        uv_hat = self._windower.remove_window(self.D_hat)

        objective = self._get_objective(z_encoder)

        # use FISTA on alternate u and v

        pobj = []
        for jj in range(1):

            # update u
            uv_hat, pobj_u = self._update_u(uv_hat, objective, z_encoder)

            # update v
            uv_hat, pobj_v = self._update_v(uv_hat, objective, z_encoder)

            if self.debug:
                pobj.extend(pobj_u)
                pobj.extend(pobj_v)

        self.D_hat = self._windower.window(uv_hat)

        z_encoder.set_D(self.D_hat)

        if self.debug:
            return self.D_hat, pobj
        return self.D_hat

    def _update_u(self, uv_hat0, objective, z_encoder):

        n_channels = self.n_channels

        uv_hat = uv_hat0.copy()
        u_hat, v_hat = uv_hat[:, :n_channels], uv_hat[:, n_channels:]

        def grad_u(u):

            uv = np.c_[u, self._windower.simple_window(v_hat)]

            grad_d = gradient_d(
                uv, X=z_encoder.X, z=None,
                constants=z_encoder.get_constants()
            )

            return (grad_d * uv[:, None, z_encoder.n_channels:]).sum(axis=2)

        def prox_u(u, step_size=None):
            u /= np.maximum(1., np.linalg.norm(u, axis=1, keepdims=True))
            return u

        def obj(u):
            uv = np.c_[u, v_hat]
            return objective(uv)

        def op_Hu(u):
            u = np.reshape(u, (self.D_hat.shape[0], n_channels))
            uv = np.c_[u, v_hat]
            H_d = numpy_convolve_uv(z_encoder.ztz, uv)
            H_u = (H_d * uv[:, None, n_channels:]).sum(axis=2)
            return H_u.ravel()

        u_hat, pobj_u = self._run_fista(u_hat, uv_hat, obj, grad_u,
                                        prox_u, op_Hu, 'u', z_encoder)

        uv_hat = np.c_[u_hat, v_hat]

        return uv_hat, pobj_u

    def _update_v(self, uv_hat0, objective, z_encoder):

        n_channels = self.n_channels

        uv_hat = uv_hat0.copy()
        u_hat, v_hat = uv_hat[:, :n_channels], uv_hat[:, n_channels:]

        def grad_v(v):

            v = self._windower.simple_window(v)

            uv = np.c_[u_hat, v]

            grad_d = gradient_d(
                uv, X=z_encoder.X, z=None,
                constants=z_encoder.get_constants()
            )

            grad_v = (grad_d * uv[:, :z_encoder.n_channels, None]).sum(axis=1)

            return self._windower.simple_window(grad_v)

        def prox_v(v, step_size=None):

            v = self._windower.simple_window(v)

            v /= np.maximum(1., np.linalg.norm(v, axis=1, keepdims=True))

            return self._windower.remove_simple_window(v)

        def obj(v):
            uv = np.c_[u_hat, v]
            return objective(uv)

        def op_Hv(v):
            v = np.reshape(v, (self.D_hat.shape[0], self.n_times_atom))
            uv = np.c_[u_hat, v]
            H_d = numpy_convolve_uv(z_encoder.ztz, uv)
            H_v = (H_d * uv[:, :n_channels, None]).sum(axis=1)
            return H_v.ravel()

        v_hat, pobj_v = self._run_fista(v_hat, uv_hat, obj, grad_v, prox_v,
                                        op_Hv, 'v', z_encoder)

        uv_hat = np.c_[u_hat, v_hat]

        return uv_hat, pobj_v

    def _run_fista(self, d_hat, uv_hat, f_obj, f_grad, f_prox, op, variable,
                   z_encoder):
        """Run FISTA for given the objective with adaptive step size or 1/L.

        Parameters
        ----------
        d_hat : array
            Initial point of the optimization (u_hat or v_hat depending on
            variable value)
        uv_hat : array, shape (n_atoms, n_channels + n_times_atom)
           The atoms
        f_obj : callable
            Objective function. Used only if debug or adaptive_step_size.
        f_grad : callable
            Gradient of the objective function
        f_prox : callable
            Proximal operator
        op : callable

        variable : str
            'u' or 'v'
        z_encoder : BaseZEncoder
            ZEncoder object.

        Returns
        -------
        x_hat : array
            The final point after optimization
        pobj : list or None
            If debug is True, pobj contains the value of the cost function at
            each iteration.

        """

        L = self._get_step_size(uv_hat, op, variable, z_encoder)

        return fista(
            f_obj, f_grad, f_prox, 0.99 / L, d_hat, self.max_iter,
            momentum=self.momentum, eps=self.eps,
            adaptive_step_size=self.adaptive_step_size, debug=self.debug,
            verbose=self.verbose, name="Update " + variable
        )

    def _get_step_size(self, uv_hat, op, variable, z_encoder):
        if self.adaptive_step_size:
            L = 1
        else:

            # compute lipschitz
            # XXX - maybe replace with scipy.sparse.linalg.svds
            b_hat_0 = np.random.randn(uv_hat.size)

            n_atoms = self.D_hat.shape[0]

            if variable == 'u':
                b_hat_0 = b_hat_0.reshape(
                    n_atoms, -1)[:, :z_encoder.n_channels].ravel()
                n_points = n_atoms * z_encoder.n_channels
                L = power_iteration(op, n_points, b_hat_0=b_hat_0)
            elif variable == 'v':
                b_hat_0 = b_hat_0.reshape(
                    n_atoms, -1)[:, z_encoder.n_channels:].ravel()
                n_points = n_atoms * self.n_times_atom

            L = power_iteration(op, n_points, b_hat_0=b_hat_0)

        assert L > 0

        return L


class DSolver(BaseDSolver):
    """A class for 'fista' solver_d when rank1 is False. """

    def __init__(self, n_channels, n_atoms, n_times_atom, solver_d,
                 uv_constraint, D_init, resample_strategy,
                 window, eps, max_iter, momentum,
                 random_state, verbose, debug):

        super().__init__(
            n_channels, n_atoms, n_times_atom, solver_d, uv_constraint, D_init,
            resample_strategy, window, eps, max_iter, momentum,
            random_state, verbose, debug
        )

        self.name = "Update D"
        self.rank1 = False

    def _init_windower(self):
        self._windower = SimpleWindower(self.n_times_atom)

    def prox(self, D_hat):
        return prox_d(D_hat)

    def get_D_shape(self):
        """Returns the expected shape of the dictionary.

        Note: For the 'greedy' strategy this does not return the actual
        dictionary shape, but the final expected shape.
        """
        return (self.n_atoms, self.n_channels, self.n_times_atom)

    def grad(self, D, z_encoder):
        return gradient_d(
            D=D, X=z_encoder.X, z=None,
            constants=z_encoder.get_constants()
        )
