import numpy as np

from .init_dict import init_dictionary
from .loss_and_gradient import compute_objective, compute_X_and_objective_multi
from .loss_and_gradient import gradient_uv, gradient_d
from .update_d_multi import prox_d, prox_uv, check_solver_and_constraints
from .utils import check_random_state
from .utils.convolution import numpy_convolve_uv
from .utils.dictionary import tukey_window, get_uv
from .utils.optim import fista, power_iteration


def get_solver_d(solver_d='alternate_adaptive',
                 rank1=False,
                 uv_constraint='auto',
                 window=False,
                 eps=1e-8,
                 max_iter=300,
                 momentum=False,
                 random_state=None):
    """Returns solver depending on solver_d type and rank1 value.

    Parameters
    ----------
    solver_d : str in {'alternate' | 'alternate_adaptive' | 'fista' | 'joint' |
    'auto'}
        The solver to use for the d update.
        - If rank1 is False, only option is 'fista'
        - If rank1 is True, options are 'alternate', 'alternate_adaptive'
          (default) or 'joint'
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
    """

    solver_d, uv_constraint = check_solver_and_constraints(rank1,
                                                           solver_d,
                                                           uv_constraint)

    if rank1:
        if solver_d in ['alternate', 'alternate_adaptive']:
            return AlternateDSolver(solver_d, rank1, uv_constraint, window,
                                    eps, max_iter, momentum, random_state)
        elif solver_d in ['fista', 'joint']:
            return JointDSolver(solver_d, rank1, uv_constraint, window,
                                eps, max_iter, momentum, random_state)
        else:
            raise ValueError('Unknown solver_d: %s' % (solver_d, ))
    else:
        if solver_d in ['fista']:
            return DSolver(solver_d, rank1, uv_constraint, window,
                           eps, max_iter, momentum, random_state)
        else:
            raise ValueError('Unknown solver_d: %s' % (solver_d, ))


class BaseDSolver:
    """Base class for a d solver."""

    def __init__(self,
                 solver_d,
                 rank1,
                 uv_constraint,
                 window,
                 eps,
                 max_iter,
                 momentum,
                 random_state):

        self.solver_d = solver_d
        self.rank1 = rank1
        self.uv_constraint = uv_constraint
        self.window = window
        self.eps = eps
        self.max_iter = max_iter
        self.momentum = momentum
        self.rng = check_random_state(random_state)

    def init_dictionary(self, X, n_atoms, n_times_atom, D_init=None,
                        D_init_params=dict()):
        """Returns an initial dictionary for the signals X.

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

        return init_dictionary(X, n_atoms, n_times_atom,
                               D_init=D_init,
                               rank1=self.rank1,
                               uv_constraint=self.uv_constraint,
                               D_init_params=D_init_params,
                               random_state=self.rng,
                               window=self.window)

    def update_D(self, z_encoder, verbose=0, debug=False):
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
        D_hat : array, shape (n_atoms, n_channels + n_times_atom) or
                             (n_atoms, n_channels, n_times_atom)
            The atoms to learn from the data.
        """
        raise NotImplementedError()


class Rank1DSolver(BaseDSolver):
    """Base class for a rank1 solver d."""

    def __init__(self,
                 solver_d,
                 rank1,
                 uv_constraint,
                 window,
                 eps,
                 max_iter,
                 momentum,
                 random_state):

        super().__init__(solver_d,
                         rank1,
                         uv_constraint,
                         window,
                         eps,
                         max_iter,
                         momentum,
                         random_state)

    def _window(self, uv_hat0, n_channels, n_times_atom):

        tukey_window_ = None

        if self.window:
            tukey_window_ = tukey_window(n_times_atom)[None, :]
            uv_hat0 = uv_hat0.copy()
            uv_hat0[:, n_channels:] /= tukey_window_

        return uv_hat0, tukey_window_

    def _objective(self, z_encoder, tukey_window_):

        def objective(uv):

            n_channels = z_encoder.n_channels

            if self.window:
                uv = uv.copy()
                uv[:, n_channels:] *= tukey_window_

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
                uv_constraint=self.uv_constraint)

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

        n_channels = z_encoder.n_channels
        n_times_atom = z_encoder.n_times_atom
        d0 = z_encoder.get_max_error_patch()

        if self.window:
            d0 = d0 * tukey_window(n_times_atom)[None, :]

        d0 = prox_uv(get_uv(d0), uv_constraint=self.uv_constraint,
                     n_channels=n_channels)

        return d0


class JointDSolver(Rank1DSolver):
    """A class for 'fista' or 'joint' solver_d when rank1 is True. """

    def __init__(self,
                 solver_d,
                 rank1,
                 uv_constraint,
                 window,
                 eps,
                 max_iter,
                 momentum,
                 random_state):

        super().__init__(solver_d,
                         rank1,
                         uv_constraint,
                         window,
                         eps,
                         max_iter,
                         momentum,
                         random_state)

    def update_D(self, z_encoder, verbose=0, debug=False):
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

        uv_hat0, tukey_window_ = self._window(z_encoder.D_hat,
                                              z_encoder.n_channels,
                                              z_encoder.n_times_atom)

        n_channels = z_encoder.n_channels

        # use FISTA on joint [u, v], with an adaptive step size

        def grad(uv):
            if self.window:
                uv = uv.copy()
                uv[:, n_channels:] *= tukey_window_

            grad = gradient_uv(uv=uv,
                               X=z_encoder.X,
                               z=z_encoder.get_z_hat(),
                               constants=z_encoder.get_constants(),
                               loss=z_encoder.loss,
                               loss_params=z_encoder.loss_params)
            if self.window:
                grad[:, n_channels:] *= tukey_window_

            return grad

        def prox(uv, step_size=None):
            if self.window:
                uv[:, n_channels:] *= tukey_window_

            uv = prox_uv(uv, uv_constraint=self.uv_constraint,
                         n_channels=n_channels)

            if self.window:
                uv[:, n_channels:] /= tukey_window_

            return uv

        objective = self._objective(z_encoder, tukey_window_)

        uv_hat, pobj = fista(objective, grad, prox, None, uv_hat0,
                             self.max_iter, momentum=self.momentum,
                             eps=self.eps, adaptive_step_size=True,
                             debug=debug, verbose=verbose, name="Update uv")

        if self.window:
            uv_hat[:, n_channels:] *= tukey_window_

        if debug:
            return uv_hat, pobj
        return uv_hat


class AlternateDSolver(Rank1DSolver):
    """A class for 'alternate' or 'alternate_adaptive' solver_d when rank1 is
       True.
    """

    def __init__(self,
                 solver_d,
                 rank1,
                 uv_constraint,
                 window,
                 eps,
                 max_iter,
                 momentum,
                 random_state):

        super().__init__(solver_d,
                         rank1,
                         uv_constraint,
                         window,
                         eps,
                         max_iter,
                         momentum,
                         random_state)

        assert self.uv_constraint == 'separate', (
            "alternate solver should be used with separate constraints"
        )

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

    def update_D(self, z_encoder, verbose=0, debug=False):
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

        uv_hat0, tukey_window_ = self._window(z_encoder.D_hat,
                                              n_channels,
                                              z_encoder.n_times_atom)

        objective = self._objective(z_encoder, tukey_window_)

        # use FISTA on alternate u and v
        adaptive_step_size = (self.solver_d == 'alternate_adaptive')

        uv_hat = uv_hat0.copy()
        u_hat, v_hat = uv_hat[:, :n_channels], uv_hat[:, n_channels:]

        def prox_u(u, step_size=None):
            u /= np.maximum(1., np.linalg.norm(u, axis=1, keepdims=True))
            return u

        def prox_v(v, step_size=None):
            if self.window:
                v *= tukey_window_

            v /= np.maximum(1., np.linalg.norm(v, axis=1, keepdims=True))

            if self.window:
                v /= tukey_window_

            return v

        pobj = []
        for jj in range(1):
            # ---------------- update u

            def obj(u):
                uv = np.c_[u, v_hat]
                return objective(uv)

            def grad_u(u):
                if self.window:
                    uv = np.c_[u, v_hat * tukey_window_]
                else:
                    uv = np.c_[u, v_hat]

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
                Lu = self.compute_lipschitz(uv_hat,
                                            z_encoder.n_channels,
                                            z_encoder.ztz,
                                            'u')
            assert Lu > 0

            u_hat, pobj_u = fista(obj, grad_u, prox_u, 0.99 / Lu, u_hat,
                                  self.max_iter, momentum=self.momentum,
                                  eps=self.eps,
                                  adaptive_step_size=adaptive_step_size,
                                  debug=debug, verbose=verbose,
                                  name="Update u")
            uv_hat = np.c_[u_hat, v_hat]
            if debug:
                pobj.extend(pobj_u)

            # ---------------- update v
            def obj(v):
                uv = np.c_[u_hat, v]
                return objective(uv)

            def grad_v(v):
                if self.window:
                    v = v * tukey_window_

                uv = np.c_[u_hat, v]
                grad_d = gradient_d(uv,
                                    X=z_encoder.X,
                                    z=z_encoder.get_z_hat(),
                                    constants=z_encoder.get_constants(),
                                    loss=z_encoder.loss,
                                    loss_params=z_encoder.loss_params)
                grad_v = (grad_d * uv[:, :n_channels, None]).sum(axis=1)

                if self.window:
                    grad_v *= tukey_window_
                return grad_v

            if adaptive_step_size:
                Lv = 1
            else:
                Lv = self.compute_lipschitz(uv_hat,
                                            z_encoder.n_channels,
                                            z_encoder.ztz,
                                            'v')
            assert Lv > 0

            v_hat, pobj_v = fista(obj, grad_v, prox_v, 0.99 / Lv, v_hat,
                                  self.max_iter, momentum=self.momentum,
                                  eps=self.eps,
                                  adaptive_step_size=adaptive_step_size,
                                  verbose=verbose, debug=debug,
                                  name="Update v")
            uv_hat = np.c_[u_hat, v_hat]
            if debug:
                pobj.extend(pobj_v)

        if self.window:
            uv_hat[:, n_channels:] *= tukey_window_

        if debug:
            return uv_hat, pobj
        return uv_hat


class DSolver(BaseDSolver):
    """A class for 'fista' solver_d when rank1 is False. """

    def __init__(self,
                 solver_d,
                 rank1,
                 uv_constraint,
                 window,
                 eps,
                 max_iter,
                 momentum,
                 random_state):

        super().__init__(solver_d,
                         rank1,
                         uv_constraint,
                         window,
                         eps,
                         max_iter,
                         momentum,
                         random_state)

    def _window(self, D_hat0, n_times_atom):
        tukey_window_ = None

        if self.window:
            tukey_window_ = tukey_window(n_times_atom)[None, None, :]
            D_hat0 = D_hat0 / tukey_window_

        return D_hat0, tukey_window_

    def _objective(self, D, z_encoder, tukey_window_, full=False):

        def objective(D, full=False):
            if self.window:
                D = D * tukey_window_

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
        n_times_atom = z_encoder.n_times_atom
        d0 = z_encoder.get_max_error_patch()

        if self.window:
            d0 = d0 * tukey_window(n_times_atom)[None, :]

        return prox_d(d0)

    def update_D(self, z_encoder, verbose=0, debug=False):
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
        D_hat : array, shape (n_atoms, n_channels, n_times_atom)
            The atoms to learn from the data.
        """

        D_hat0, tukey_window_ = self._window(
            z_encoder.D_hat, z_encoder.n_times_atom)

        objective = self._objective(D_hat0, z_encoder, tukey_window_)

        def grad(D):
            if self.window:
                D = D * tukey_window_

            grad = gradient_d(D=D,
                              X=z_encoder.X,
                              z=z_encoder.get_z_hat(),
                              constants=z_encoder.get_constants(),
                              loss=z_encoder.loss,
                              loss_params=z_encoder.loss_params)
            if self.window:
                grad *= tukey_window_

            return grad

        def prox(D, step_size=None):
            if self.window:
                D *= tukey_window_
            D = prox_d(D)
            if self.window:
                D /= tukey_window_
            return D

        D_hat, pobj = fista(objective, grad, prox, None, D_hat0, self.max_iter,
                            verbose=verbose, momentum=self.momentum,
                            eps=self.eps, adaptive_step_size=True, debug=debug,
                            name="Update D")

        if self.window:
            D_hat = D_hat * tukey_window_

        if debug:
            return D_hat, pobj
        return D_hat
