import numpy as np

from .init_dict import init_dictionary
from .loss_and_gradient import gradient_uv, gradient_d
from .update_d_multi import prox_d, prox_uv, check_solver_and_constraints
from .utils import check_random_state
from .utils.convolution import numpy_convolve_uv
from .utils.dictionary import NoWindow, UVWindower, SimpleWindower, get_uv
from .utils.optim import fista, power_iteration


def get_solver_d(solver_d='alternate_adaptive', rank1=False,
                 uv_constraint='auto', window=False, eps=1e-8,
                 max_iter=300, momentum=False, random_state=None,
                 verbose=0, debug=False):
    """Returns solver depending on solver_d type and rank1 value.

    Parameters
    ----------
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
                solver_d, uv_constraint, window, eps, max_iter, momentum,
                random_state, verbose, debug
            )
        elif solver_d in ['fista', 'joint']:
            return JointDSolver(
                solver_d, uv_constraint, window, eps, max_iter, momentum,
                random_state, verbose, debug
            )
        else:
            raise ValueError('Unknown solver_d: %s' % (solver_d, ))
    else:
        if solver_d in ['auto', 'fista']:
            return DSolver(
                solver_d, uv_constraint, window, eps, max_iter, momentum,
                random_state, verbose, debug
            )
        else:
            raise ValueError('Unknown solver_d: %s' % (solver_d, ))


class BaseDSolver:
    """Base class for a d solver."""

    def __init__(self, solver_d, uv_constraint, window,
                 eps, max_iter, momentum, random_state, verbose,
                 debug, rank1):

        self.rank1 = rank1
        self.uv_constraint = uv_constraint
        self.window = window
        self.eps = eps
        self.max_iter = max_iter
        self.momentum = momentum
        self.rng = check_random_state(random_state)
        self.verbose = verbose
        self.debug = debug

        self.windower = None

        # This guarantees that, if self.window=False, self.windower=NoWindow()
        if not self.window:
            self.windower = NoWindow()

    def init_dictionary(self, X, n_atoms, n_times_atom, D_init=None,
                        D_init_params=dict()):
        """Returns an initial dictionary for the signal X.

        Parameter
        ---------
        X: array, shape (n_trials, n_channels, n_times)
            The data on which to perform CSC.
        n_atoms : int
            The number of atoms to learn.
        n_times_atom : int
            The support of the atom.
        D_init : array or {'kmeans' | 'ssa' | 'chunk' | 'random'}
            The initialization scheme for the dictionary or the initial
            atoms. The shape should match the required dictionary shape, ie if
            rank1 is True, (n_atoms, n_channels + n_times_atom) and else
            (n_atoms, n_channels, n_times_atom)
        D_init_params : dict
            Dictionnary of parameters for the kmeans init method.

        Return
        ------
        D : array shape (n_atoms, n_channels + n_times_atom) or
                  shape (n_atoms, n_channels, n_times_atom)
            The initial atoms to learn from the data.
        """

        return init_dictionary(
            X, n_atoms, n_times_atom, D_init=D_init,
            rank1=self.rank1, uv_constraint=self.uv_constraint,
            D_init_params=D_init_params, random_state=self.rng,
            window=self.window
        )

    def update_D(self, z_encoder):
        """Learn d's in time domain.

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

        self._init_windower(z_encoder)

        D_hat0 = self.windower.dewindow(z_encoder.D_hat)

        D_hat, pobj = fista(
            self._get_objective(z_encoder), self._get_grad(z_encoder),
            self._get_prox(z_encoder), None, D_hat0, self.max_iter,
            momentum=self.momentum, eps=self.eps, adaptive_step_size=True,
            name=self.name, debug=self.debug, verbose=self.verbose
        )

        D_hat = self.windower.window(D_hat)

        if self.debug:
            return D_hat, pobj
        return D_hat

    def get_max_error_dict(self, z_encoder):
        """Get the maximal reconstruction error patch from the data as a new atom

        This idea is used for instance in [Yellin2017]

        Parameters
        ----------
        z_encoder : BaseZEncoder
            ZEncoder object to be able to compute the largest error patch.

        Return
        ------
        dk: array, shape (n_channels + n_times_atom,) or
                         (n_channels, n_times_atom,)
            New atom for the dictionary, chosen as the chunk of data with the
            maximal reconstruction error.

        [Yellin2017] BLOOD CELL DETECTION AND COUNTING IN HOLOGRAPHIC LENS-FREE
        IMAGING BY CONVOLUTIONAL SPARSE DICTIONARY LEARNING AND CODING.
        """

        raise NotImplementedError()

    def _init_windower(self, z_encoder):
        raise NotImplementedError()

    def _get_objective(self, z_encoder):

        def objective(D, full=False):

            D = self.windower.window(D)

            return z_encoder.compute_objective(D)

        return objective


class Rank1DSolver(BaseDSolver):
    """Base class for a rank1 solver d."""

    def __init__(self, solver_d, uv_constraint, window, eps, max_iter,
                 momentum, random_state, verbose, debug):

        super().__init__(
            solver_d, uv_constraint, window, eps, max_iter, momentum,
            random_state, verbose, debug, rank1=True
        )

        self.name = "Update uv"

    def get_max_error_dict(self, z_encoder):
        """Get the maximal reconstruction error patch from the data as a new atom

        This idea is used for instance in [Yellin2017]

        Parameters
        ----------
        z_encoder : BaseZEncoder
            ZEncoder object to be able to compute the largest error patch.

        Return
        ------
        uvk: array, shape (n_channels + n_times_atom,)
            New atom for the dictionary, chosen as the chunk of data with the
            maximal reconstruction error.

        [Yellin2017] BLOOD CELL DETECTION AND COUNTING IN HOLOGRAPHIC LENS-FREE
        IMAGING BY CONVOLUTIONAL SPARSE DICTIONARY LEARNING AND CODING.
        """
        self._init_windower(z_encoder)

        d0 = z_encoder.get_max_error_patch()

        d0 = self.windower.simple_window(d0)

        return prox_uv(get_uv(d0), uv_constraint=self.uv_constraint,
                       n_channels=z_encoder.n_channels)

    def _init_windower(self, z_encoder):
        if self.windower is None:
            self.windower = UVWindower(z_encoder.n_times_atom,
                                       z_encoder.n_channels)


class JointDSolver(Rank1DSolver):
    """A class for 'fista' or 'joint' solver_d when rank1 is True. """

    def __init__(self, solver_d, uv_constraint, window, eps, max_iter,
                 momentum, random_state, verbose, debug):

        super().__init__(
            solver_d, uv_constraint, window, eps, max_iter,
            momentum, random_state, verbose, debug
        )

    def _get_grad(self, z_encoder):

        def grad(uv):

            uv = self.windower.window(uv)

            grad = gradient_uv(
                uv=uv, X=z_encoder.X, z=z_encoder.get_z_hat(),
                constants=z_encoder.get_constants(), loss=z_encoder.loss,
                loss_params=z_encoder.loss_params
            )

            return self.windower.window(grad)

        return grad

    def _get_prox(self, z_encoder):

        def prox(uv, step_size=None):

            uv = self.windower.window(uv)

            uv = prox_uv(uv, uv_constraint=self.uv_constraint,
                         n_channels=z_encoder.n_channels)

            return self.windower.dewindow(uv)

        return prox


class AlternateDSolver(Rank1DSolver):
    """A class for 'alternate' or 'alternate_adaptive' solver_d when rank1 is
       True.
    """

    def __init__(self, solver_d, uv_constraint, window, eps, max_iter,
                 momentum, random_state, verbose, debug):

        super().__init__(
            solver_d, uv_constraint, window, eps, max_iter,
            momentum, random_state, verbose, debug
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

        self._init_windower(z_encoder)

        uv_hat = self.windower.dewindow(z_encoder.D_hat)

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

        uv_hat = self.windower.window(uv_hat)

        if self.debug:
            return uv_hat, pobj
        return uv_hat

    def _update_u(self, uv_hat0, objective, z_encoder):

        n_channels = z_encoder.n_channels

        uv_hat = uv_hat0.copy()
        u_hat, v_hat = uv_hat[:, :n_channels], uv_hat[:, n_channels:]

        def grad_u(u):

            uv = np.c_[u, self.windower.simple_window(v_hat)]

            grad_d = gradient_d(
                uv, X=z_encoder.X, z=z_encoder.get_z_hat(),
                constants=z_encoder.get_constants(), loss=z_encoder.loss,
                loss_params=z_encoder.loss_params
            )

            return (grad_d * uv[:, None, z_encoder.n_channels:]).sum(axis=2)

        def prox_u(u, step_size=None):
            u /= np.maximum(1., np.linalg.norm(u, axis=1, keepdims=True))
            return u

        def obj(u):
            uv = np.c_[u, v_hat]
            return objective(uv)

        def op_Hu(u):
            u = np.reshape(u, (z_encoder.n_atoms, n_channels))
            uv = np.c_[u, v_hat]
            H_d = numpy_convolve_uv(z_encoder.ztz, uv)
            H_u = (H_d * uv[:, None, n_channels:]).sum(axis=2)
            return H_u.ravel()

        u_hat, pobj_u = self._run_fista(u_hat, uv_hat, obj, grad_u,
                                        prox_u, op_Hu, 'u', z_encoder)

        uv_hat = np.c_[u_hat, v_hat]

        return uv_hat, pobj_u

    def _update_v(self, uv_hat0, objective, z_encoder):

        n_channels = z_encoder.n_channels

        uv_hat = uv_hat0.copy()
        u_hat, v_hat = uv_hat[:, :n_channels], uv_hat[:, n_channels:]

        def grad_v(v):

            v = self.windower.simple_window(v)

            uv = np.c_[u_hat, v]

            grad_d = gradient_d(
                uv, X=z_encoder.X, z=z_encoder.get_z_hat(),
                constants=z_encoder.get_constants(), loss=z_encoder.loss,
                loss_params=z_encoder.loss_params
            )

            grad_v = (grad_d * uv[:, :z_encoder.n_channels, None]).sum(axis=1)

            return self.windower.simple_window(grad_v)

        def prox_v(v, step_size=None):

            v = self.windower.simple_window(v)

            v /= np.maximum(1., np.linalg.norm(v, axis=1, keepdims=True))

            return self.windower.simple_dewindow(v)

        def obj(v):
            uv = np.c_[u_hat, v]
            return objective(uv)

        def op_Hv(v):
            v = np.reshape(v, (z_encoder.n_atoms, z_encoder.n_times_atom))
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
            if z_encoder.loss != 'l2':
                raise NotImplementedError()

            # compute lipschitz
            # XXX - maybe replace with scipy.sparse.linalg.svds
            b_hat_0 = np.random.randn(uv_hat.size)

            if variable == 'u':
                b_hat_0 = b_hat_0.reshape(
                    z_encoder.n_atoms, -1)[:, :z_encoder.n_channels].ravel()
                n_points = z_encoder.n_atoms * z_encoder.n_channels
                L = power_iteration(op, n_points, b_hat_0=b_hat_0)
            elif variable == 'v':
                b_hat_0 = b_hat_0.reshape(
                    z_encoder.n_atoms, -1)[:, z_encoder.n_channels:].ravel()
                n_points = z_encoder.n_atoms * z_encoder.n_times_atom

            L = power_iteration(op, n_points, b_hat_0=b_hat_0)

        assert L > 0

        return L


class DSolver(BaseDSolver):
    """A class for 'fista' solver_d when rank1 is False. """

    def __init__(self, solver_d, uv_constraint, window, eps, max_iter,
                 momentum, random_state, verbose, debug):

        super().__init__(
            solver_d, uv_constraint, window, eps, max_iter, momentum,
            random_state, verbose, debug, rank1=False
        )

        self.name = "Update D"

    def get_max_error_dict(self, z_encoder):
        """Get the maximal reconstruction error patch from the data as a new atom

        This idea is used for instance in [Yellin2017]

        Parameters
        ----------
        z_encoder : BaseZEncoder
            ZEncoder object to be able to compute the largest error patch.

        Return
        ------
        dk: array, shape (n_channels, n_times_atom,)
            New atom for the dictionary, chosen as the chunk of data with the
            maximal reconstruction error.

        [Yellin2017] BLOOD CELL DETECTION AND COUNTING IN HOLOGRAPHIC LENS-FREE
        IMAGING BY CONVOLUTIONAL SPARSE DICTIONARY LEARNING AND CODING.
        """

        self._init_windower(z_encoder)

        d0 = z_encoder.get_max_error_patch()

        d0 = self.windower.window(d0)

        return prox_d(d0)

    def _init_windower(self, z_encoder):
        if self.windower is None:
            self.windower = SimpleWindower(z_encoder.n_times_atom)

    def _get_grad(self, z_encoder):

        def grad(D):

            D = self.windower.window(D)

            grad = gradient_d(
                D=D, X=z_encoder.X, z=z_encoder.get_z_hat(),
                constants=z_encoder.get_constants(), loss=z_encoder.loss,
                loss_params=z_encoder.loss_params
            )

            return self.windower.window(grad)

        return grad

    def _get_prox(self, z_encoder):

        def prox(D, step_size=None):

            D = self.windower.window(D)

            D = prox_d(D)

            return self.windower.dewindow(D)

        return prox
