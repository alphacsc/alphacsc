import numpy as np

from .loss_and_gradient import compute_objective, compute_X_and_objective_multi
from .loss_and_gradient import gradient_uv, gradient_d

from .utils import check_random_state
from .utils.convolution import numpy_convolve_uv
from .utils.dictionary import tukey_window
from .utils.optim import fista, power_iteration


def get_solver_d(solver_d='alternate_adaptive',
                 rank1=False,
                 uv_constraint='auto',
                 window=False,
                 eps=1e-8,
                 max_iter=300,
                 momentum=False,
                 random_state=None):

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
        if solver_d in ['fista', 'auto']:
            return DSolver(solver_d, rank1, uv_constraint, window,
                           eps, max_iter, momentum, random_state)
        else:
            raise ValueError('Unknown solver_d: %s' % (solver_d, ))


class BaseDSolver:

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

    def squeeze_all_except_one(self, X, axis=0):
        squeeze_axis = tuple(set(range(X.ndim)) - set([axis]))
        return X.squeeze(axis=squeeze_axis)

    def update_D(self, z_encoder, verbose=0, debug=False):
        raise NotImplementedError()


class Rank1DSolver(BaseDSolver):

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


class JointDSolver(Rank1DSolver):

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

    def prox_uv(self, uv, n_channels=None,
                return_norm=False):

        if self.uv_constraint == 'joint':
            norm_uv = np.maximum(1, np.linalg.norm(uv, axis=1, keepdims=True))
            uv /= norm_uv

        elif self.uv_constraint == 'separate':
            assert n_channels is not None
            norm_u = np.maximum(1, np.linalg.norm(uv[:, :n_channels],
                                                  axis=1, keepdims=True))
            norm_v = np.maximum(1, np.linalg.norm(uv[:, n_channels:],
                                                  axis=1, keepdims=True))

            uv[:, :n_channels] /= norm_u
            uv[:, n_channels:] /= norm_v
            norm_uv = norm_u * norm_v
        else:
            raise ValueError('Unknown uv_constraint: %s.' %
                             (self.uv_constraint, ))

        if return_norm:
            return uv, self.squeeze_all_except_one(norm_uv, axis=0)
        else:
            return uv

    def update_D(self, z_encoder, verbose=0, debug=False):

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

            uv = self.prox_uv(uv, n_channels=n_channels)

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
        if self.b_hat_0 is None:
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

    def prox_d(self, D, return_norm=False):
        norm_d = np.maximum(1, np.linalg.norm(D, axis=(1, 2), keepdims=True))
        D /= norm_d

        if return_norm:
            return D, self.squeeze_all_except_one(norm_d, axis=0)
        else:
            return D

    def update_D(self, z_encoder, verbose=0, debug=False):

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
            D = self.prox_d(D)
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
