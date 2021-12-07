import numpy as np

from .init_dict import init_dictionary
from .loss_and_gradient import compute_objective, compute_X_and_objective_multi
from .loss_and_gradient import gradient_uv, gradient_d
from .update_d_multi import prox_d, prox_uv
from .utils import check_random_state
from .utils.convolution import numpy_convolve_uv
from .utils.dictionary import tukey_window, get_uv
from .utils.optim import fista, power_iteration


def get_solver_d(n_times_atom,
                 solver_d='alternate_adaptive',
                 rank1=False,
                 window=False,
                 eps=1e-8,
                 max_iter=300,
                 momentum=False,
                 random_state=None,
                 verbose=0,
                 debug=False):
    """Returns solver depending on solver_d type and rank1 value.

    Parameters
    ----------
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

    if rank1:
        if solver_d in ['auto', 'alternate', 'alternate_adaptive']:
            return AlternateDSolver(n_times_atom, solver_d, window, eps,
                                    max_iter, momentum, random_state, verbose,
                                    debug)
        elif solver_d in ['fista', 'joint']:
            return JointDSolver(n_times_atom, solver_d, window, eps, max_iter,
                                momentum, random_state, verbose, debug)
        else:
            raise ValueError('Unknown solver_d: %s' % (solver_d, ))
    else:
        if solver_d in ['auto', 'fista']:
            return DSolver(n_times_atom, solver_d, window, eps, max_iter,
                           momentum, random_state, verbose, debug)
        else:
            raise ValueError('Unknown solver_d: %s' % (solver_d, ))


class BaseDSolver:
    """Base class for a d solver."""

    def __init__(self,
                 n_times_atom,
                 solver_d,
                 window,
                 eps,
                 max_iter,
                 momentum,
                 random_state,
                 verbose,
                 debug):

        self.n_times_atom = n_times_atom
        self.solver_d = solver_d
        self.window = window
        self.eps = eps
        self.max_iter = max_iter
        self.momentum = momentum
        self.rng = check_random_state(random_state)
        self.verbose = verbose
        self.debug = debug

        self._set_tukey_window()

    def init_dictionary(self, X, n_atoms, uv_constraint='auto', D_init=None,
                        D_init_params=dict()):
        """Returns an initial dictionary for the signal X.

        Parameter
        ---------
        X: array, shape (n_trials, n_channels, n_times)
            The data on which to perform CSC.
        n_atoms : int
            The number of atoms to learn.
        uv_constraint : str in {'joint' | 'separate' | 'auto'}
            The kind of norm constraint on the atoms if using rank1=True.
            If 'joint', the constraint is norm_2([u, v]) <= 1
            If 'separate', the constraint is norm_2(u) <= 1 and norm_2(v) <= 1
            If rank1 is False, then uv_constraint must be 'auto'.
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

        return init_dictionary(X, n_atoms, self.n_times_atom,
                               D_init=D_init,
                               rank1=self.rank1,
                               uv_constraint=uv_constraint,
                               D_init_params=D_init_params,
                               random_state=self.rng,
                               window=self.window)

    def _set_tukey_window(self, n_times_atom):
        raise NotImplementedError()

    def _window(self, d):
        if self.window:
            return d * self.tukey_window
        return d

    def _dewindow(self, d):
        if self.window:
            return d / self.tukey_window
        return d

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
        raise NotImplementedError()

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

        d0 = z_encoder.get_max_error_patch()

        return self._window(d0)


class Rank1DSolver(BaseDSolver):
    """Base class for a rank1 solver d."""

    def __init__(self,
                 n_times_atom,
                 solver_d,
                 window,
                 eps,
                 max_iter,
                 momentum,
                 random_state,
                 verbose,
                 debug):

        super().__init__(n_times_atom,
                         solver_d,
                         window,
                         eps,
                         max_iter,
                         momentum,
                         random_state,
                         verbose,
                         debug)
        self.rank1 = True

    def _set_tukey_window(self):

        if self.window:
            self.tukey_window = tukey_window(self.n_times_atom)[None, :]

    def _objective(self, z_encoder):

        def objective(uv):

            n_channels = z_encoder.n_channels

            uv = uv.copy()
            uv[:, n_channels:] = self._window(uv[:, n_channels:])

            if z_encoder.loss == 'l2':
                return compute_objective(D=uv,
                                         constants=z_encoder.get_constants())

            return compute_X_and_objective_multi(
                z_encoder.X,
                z_encoder.get_z_hat(),
                D_hat=uv,
                loss=z_encoder.loss,
                loss_params=z_encoder.loss_params,
                feasible_evaluation=z_encoder.feasible_evaluation,
                uv_constraint=z_encoder.uv_constraint
            )

        return objective

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

        d0 = super().get_max_error_dict(z_encoder)

        d0 = prox_uv(get_uv(d0), uv_constraint=z_encoder.uv_constraint,
                     n_channels=z_encoder.n_channels)

        return d0


class JointDSolver(Rank1DSolver):
    """A class for 'fista' or 'joint' solver_d when rank1 is True. """

    def __init__(self,
                 n_times_atom,
                 solver_d,
                 window,
                 eps,
                 max_iter,
                 momentum,
                 random_state,
                 verbose,
                 debug):

        super().__init__(n_times_atom,
                         solver_d,
                         window,
                         eps,
                         max_iter,
                         momentum,
                         random_state,
                         verbose,
                         debug)

    def update_D(self, z_encoder):
        """Learn d's in time domain.

        Parameters
        ----------
        z_encoder: BaseZEncoder
            ZEncoder object.
        verbose: int
            Verbosity level.
        debug : bool
            If True, return the cost at each iteration.

        Returns
        -------
        uv_hat : array, shape (n_atoms, n_channels + n_times_atom)
            The atoms to learn from the data.
        """

        n_channels = z_encoder.n_channels
        uv_hat0 = z_encoder.D_hat.copy()

        uv_hat0[:, n_channels:] = self._dewindow(uv_hat0[:, n_channels:])

        # use FISTA on joint [u, v], with an adaptive step size

        def grad(uv):

            uv = uv.copy()

            uv[:, n_channels:] = self._window(uv[:, n_channels:])

            grad = gradient_uv(uv=uv,
                               X=z_encoder.X,
                               z=z_encoder.get_z_hat(),
                               constants=z_encoder.get_constants(),
                               loss=z_encoder.loss,
                               loss_params=z_encoder.loss_params)

            grad[:, n_channels:] = self._window(grad[:, n_channels:])

            return grad

        def prox(uv, step_size=None):

            uv[:, n_channels:] = self._window(uv[:, n_channels:])

            uv = prox_uv(uv, uv_constraint=z_encoder.uv_constraint,
                         n_channels=n_channels)

            uv[:, n_channels:] = self._dewindow(uv[:, n_channels:])

            return uv

        objective = self._objective(z_encoder)

        uv_hat, pobj = fista(objective, grad, prox, None, uv_hat0,
                             self.max_iter, momentum=self.momentum,
                             eps=self.eps, adaptive_step_size=True,
                             debug=self.debug, verbose=self.verbose,
                             name="Update uv")

        uv_hat[:, n_channels:] = self._window(uv_hat[:, n_channels:])

        if self.debug:
            return uv_hat, pobj
        return uv_hat


class AlternateDSolver(Rank1DSolver):
    """A class for 'alternate' or 'alternate_adaptive' solver_d when rank1 is
       True.
    """

    def __init__(self,
                 n_times_atom,
                 solver_d,
                 window,
                 eps,
                 max_iter,
                 momentum,
                 random_state,
                 verbose,
                 debug):

        super().__init__(n_times_atom,
                         solver_d,
                         window,
                         eps,
                         max_iter,
                         momentum,
                         random_state,
                         verbose,
                         debug)

    def compute_lipschitz(self, uv0, n_channels, ztz, variable):

        u0, v0 = uv0[:, :n_channels], uv0[:, n_channels:]
        n_atoms = uv0.shape[0]
        n_times_atom = uv0.shape[1] - n_channels
        if not hasattr(self, 'b_hat_0'):
            self.b_hat_0 = np.random.randn(uv0.size)

        def op_Hu(u):
            u = np.reshape(u, (n_atoms, n_channels))
            uv = np.c_[u, v0]
            H_d = numpy_convolve_uv(ztz, uv)
            H_u = (H_d * uv[:, None, n_channels:]).sum(axis=2)
            return H_u.ravel()

        def op_Hv(v):
            v = np.reshape(v, (n_atoms, n_times_atom))
            uv = np.c_[u0, v]
            H_d = numpy_convolve_uv(ztz, uv)
            H_v = (H_d * uv[:, :n_channels, None]).sum(axis=1)
            return H_v.ravel()

        if variable == 'u':
            b_hat_u0 = self.b_hat_0.reshape(
                n_atoms, -1)[:, :n_channels].ravel()
            n_points = n_atoms * n_channels
            L = power_iteration(op_Hu, n_points, b_hat_0=b_hat_u0)
        elif variable == 'v':
            b_hat_v0 = self.b_hat_0.reshape(
                n_atoms, -1)[:, n_channels:].ravel()
            n_points = n_atoms * n_times_atom
            L = power_iteration(op_Hv, n_points, b_hat_0=b_hat_v0)
        else:
            raise ValueError("variable should be either 'u' or 'v'")
        return L

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

        n_channels = z_encoder.n_channels
        uv_hat0 = z_encoder.D_hat.copy()

        uv_hat0[:, n_channels:] = self._dewindow(uv_hat0[:, n_channels:])

        objective = self._objective(z_encoder)

        # use FISTA on alternate u and v
        adaptive_step_size = (self.solver_d == 'alternate_adaptive')

        uv_hat = uv_hat0.copy()
        u_hat, v_hat = uv_hat[:, :n_channels], uv_hat[:, n_channels:]

        def prox_u(u, step_size=None):
            u /= np.maximum(1., np.linalg.norm(u, axis=1, keepdims=True))
            return u

        def prox_v(v, step_size=None):

            v = self._window(v)

            v /= np.maximum(1., np.linalg.norm(v, axis=1, keepdims=True))

            return self._dewindow(v)

        pobj = []
        for jj in range(1):
            # ---------------- update u

            def obj(u):
                uv = np.c_[u, v_hat]
                return objective(uv)

            def grad_u(u):

                uv = np.c_[u, self._window(v_hat)]

                grad_d = gradient_d(uv,
                                    X=z_encoder.X,
                                    z=z_encoder.get_z_hat(),
                                    constants=z_encoder.get_constants(),
                                    loss=z_encoder.loss,
                                    loss_params=z_encoder.loss_params)

                return (grad_d * uv[:, None, n_channels:]).sum(axis=2)

            if adaptive_step_size:
                Lu = 1
            else:
                if z_encoder.loss != 'l2':
                    raise NotImplementedError()
                Lu = self.compute_lipschitz(uv_hat,
                                            z_encoder.n_channels,
                                            z_encoder.ztz,
                                            'u')
            assert Lu > 0

            u_hat, pobj_u = fista(obj, grad_u, prox_u, 0.99 / Lu, u_hat,
                                  self.max_iter, momentum=self.momentum,
                                  eps=self.eps,
                                  adaptive_step_size=adaptive_step_size,
                                  debug=self.debug, verbose=self.verbose,
                                  name="Update u")
            uv_hat = np.c_[u_hat, v_hat]
            if self.debug:
                pobj.extend(pobj_u)

            # ---------------- update v
            def obj(v):
                uv = np.c_[u_hat, v]
                return objective(uv)

            def grad_v(v):

                v = self._window(v)

                uv = np.c_[u_hat, v]
                grad_d = gradient_d(uv,
                                    X=z_encoder.X,
                                    z=z_encoder.get_z_hat(),
                                    constants=z_encoder.get_constants(),
                                    loss=z_encoder.loss,
                                    loss_params=z_encoder.loss_params)
                grad_v = (grad_d * uv[:, :n_channels, None]).sum(axis=1)

                return self._window(grad_v)

            if adaptive_step_size:
                Lv = 1
            else:
                if z_encoder.loss != 'l2':
                    raise NotImplementedError()
                Lv = self.compute_lipschitz(uv_hat,
                                            z_encoder.n_channels,
                                            z_encoder.ztz,
                                            'v')
            assert Lv > 0

            v_hat, pobj_v = fista(obj, grad_v, prox_v, 0.99 / Lv, v_hat,
                                  self.max_iter, momentum=self.momentum,
                                  eps=self.eps,
                                  adaptive_step_size=adaptive_step_size,
                                  verbose=self.verbose, debug=self.debug,
                                  name="Update v")
            uv_hat = np.c_[u_hat, v_hat]
            if self.debug:
                pobj.extend(pobj_v)

        uv_hat[:, n_channels:] = self._window(uv_hat[:, n_channels:])

        if self.debug:
            return uv_hat, pobj
        return uv_hat


class DSolver(BaseDSolver):
    """A class for 'fista' solver_d when rank1 is False. """

    def __init__(self,
                 n_times_atom,
                 solver_d,
                 window,
                 eps,
                 max_iter,
                 momentum,
                 random_state,
                 verbose,
                 debug):

        super().__init__(n_times_atom,
                         solver_d,
                         window,
                         eps,
                         max_iter,
                         momentum,
                         random_state,
                         verbose,
                         debug)

        self.rank1 = False

    def _set_tukey_window(self):

        if self.window:
            self.tukey_window = tukey_window(self.n_times_atom)[None, None, :]

    def _objective(self, D, z_encoder, full=False):

        def objective(D, full=False):

            D = self._window(D)

            if z_encoder.loss == 'l2':
                return compute_objective(D=D,
                                         constants=z_encoder.get_constants())

            return compute_X_and_objective_multi(z_encoder.X,
                                                 z_encoder.get_z_hat(),
                                                 D_hat=D,
                                                 loss=z_encoder.loss,
                                                 loss_params=z_encoder.loss_params)  # noqa

        return objective

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

        d0 = super().get_max_error_dict(z_encoder)

        return prox_d(d0)

    def update_D(self, z_encoder):
        """Learn d's in time domain.

        Parameters
        ----------
        z_encoder: BaseZEncoder
            ZEncoder object.

        Returns
        -------
        D_hat : array, shape (n_atoms, n_channels, n_times_atom)
            The atoms to learn from the data.
        """

        D_hat0 = self._dewindow(z_encoder.D_hat)

        objective = self._objective(D_hat0, z_encoder)

        def grad(D):

            D = self._window(D)

            grad = gradient_d(D=D,
                              X=z_encoder.X,
                              z=z_encoder.get_z_hat(),
                              constants=z_encoder.get_constants(),
                              loss=z_encoder.loss,
                              loss_params=z_encoder.loss_params)

            return self._window(grad)

        def prox(D, step_size=None):

            D = self._window(D)

            D = prox_d(D)

            return self._dewindow(D)

        D_hat, pobj = fista(objective, grad, prox, None, D_hat0, self.max_iter,
                            verbose=self.verbose, momentum=self.momentum,
                            eps=self.eps, adaptive_step_size=True,
                            debug=self.debug, name="Update D")

        D_hat = self._window(D_hat)

        if self.debug:
            return D_hat, pobj
        return D_hat
