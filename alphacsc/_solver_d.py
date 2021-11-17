import numpy as np

from .loss_and_gradient import compute_objective, compute_X_and_objective_multi
from .loss_and_gradient import gradient_uv, gradient_d

from .update_d_multi import prox_uv, prox_d, compute_lipschitz
from .utils.dictionary import tukey_window
from .utils.optim import fista


def get_solver_d(solver_d='alternate_adaptive',
                 rank1=False,
                 window=False,
                 b_hat_0=None,
                 eps=1e-8,
                 max_iter=300,
                 momentum=False):

    if rank1:
        if solver_d in ['alternate', 'alternate_adaptive']:
            return AlternateDSolver(solver_d, rank1, window, b_hat_0, eps,
                                    max_iter, momentum)
        elif solver_d in ['fista', 'joint']:
            return JointDSolver(solver_d, rank1, window, b_hat_0, eps,
                                max_iter, momentum)
        else:
            raise ValueError('Unknown solver_d: %s' % (solver_d, ))
    else:
        if solver_d in ['fista', 'auto']:
            return DSolver(solver_d, rank1, window, b_hat_0, eps, max_iter,
                           momentum)
        else:
            raise ValueError('Unknown solver_d: %s' % (solver_d, ))


class BaseDSolver:

    def __init__(self,
                 solver_d,
                 rank1,
                 window,
                 b_hat_0,
                 eps,
                 max_iter,
                 momentum):

        self.solver_d = solver_d
        self.rank1 = rank1
        self.window = window
        self.b_hat_0 = b_hat_0
        self.eps = eps
        self.max_iter = max_iter
        self.momentum = momentum

    def update_D(self, z_encoder, verbose=0, debug=False):
        raise NotImplementedError()


class Rank1DSolver(BaseDSolver):

    def __init__(self,
                 solver_d,
                 rank1,
                 window,
                 b_hat_0,
                 eps,
                 max_iter,
                 momentum):

        super().__init__(solver_d,
                         rank1,
                         window,
                         b_hat_0,
                         eps,
                         max_iter,
                         momentum)

    def _window(self, uv_hat0, n_channels, n_times_atom):

        tukey_window_ = None

        if self.window:
            tukey_window_ = tukey_window(n_times_atom)[None, :]
            uv_hat0 = uv_hat0.copy()
            uv_hat0[:, n_channels:] /= tukey_window_

        return uv_hat0, tukey_window_

    def _objective(self, z_encoder, tukey_window_):

        def objective(uv):

            if self.window:
                uv = uv.copy()
                uv[:, z_encoder.n_channels:] *= tukey_window_

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
                uv_constraint=z_encoder.uv_constraint)

        return objective


class JointDSolver(Rank1DSolver):

    def __init__(self,
                 solver_d,
                 rank1,
                 window,
                 b_hat_0,
                 eps,
                 max_iter,
                 momentum):

        super().__init__(solver_d,
                         rank1,
                         window,
                         b_hat_0,
                         eps,
                         max_iter,
                         momentum)

    def update_D(self, z_encoder, verbose=0, debug=False):
        n_channels = z_encoder.n_channels

        uv_hat0, tukey_window_ = self._window(z_encoder.D_hat,
                                              n_channels,
                                              z_encoder.n_times_atom)

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

            uv = prox_uv(uv,
                         uv_constraint=z_encoder.uv_constraint,
                         n_channels=n_channels)

            if self.window:
                uv[:, n_channels:] /= tukey_window_

            return uv

        objective = self._objective(z_encoder, tukey_window_)

        uv_hat, pobj = fista(objective, grad, prox, None, uv_hat0,
                             self.max_iter,
                             momentum=self.momentum, eps=self.eps,
                             adaptive_step_size=True, debug=debug,
                             verbose=verbose, name="Update uv")

        if self.window:
            uv_hat[:, n_channels:] *= tukey_window_

        if debug:
            return uv_hat, pobj
        return uv_hat


class AlternateDSolver(Rank1DSolver):

    def __init__(self,
                 solver_d,
                 rank1,
                 window,
                 b_hat_0,
                 eps,
                 max_iter,
                 momentum):

        super().__init__(solver_d,
                         rank1,
                         window,
                         b_hat_0,
                         eps,
                         max_iter,
                         momentum)

    def update_D(self, z_encoder, verbose=0, debug=False):
        assert z_encoder.uv_constraint == 'separate', (
            "alternate solver should be used with separate constraints"
        )

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
                Lu = compute_lipschitz(uv_hat,
                                       z_encoder.get_constants(),
                                       'u',
                                       self.b_hat_0)
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
                Lv = compute_lipschitz(uv_hat,
                                       z_encoder.get_constants(),
                                       'v',
                                       self.b_hat_0)
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
                 window,
                 b_hat_0,
                 eps,
                 max_iter,
                 momentum):

        super().__init__(solver_d,
                         rank1,
                         window,
                         b_hat_0,
                         eps,
                         max_iter,
                         momentum)

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
