import pytest

import numpy as np

from alphacsc.utils import check_random_state
from alphacsc.utils.compute_constants import compute_ztz, compute_ztX
from alphacsc.loss_and_gradient import (
    compute_objective, compute_X_and_objective_multi
)

N_TRIALS, N_CHANNELS, N_TIMES = 5, 3, 100
N_TIMES_ATOM, N_ATOMS = 10, 4

parametrize_solver_and_constraint = pytest.mark.parametrize(
    'rank1, solver_d, uv_constraint',
    [
        (True, 'auto', 'auto'),
        (False, 'auto', 'auto'),
        (False, 'fista', 'auto'),
        (True, 'joint', 'auto'),
        (True, 'joint', 'joint'),
        (True, 'joint', 'separate'),
        (True, 'fista', 'auto'),
        (True, 'fista', 'joint'),
        (True, 'fista', 'separate'),
        (True, 'alternate_adaptive', 'separate')
    ]
)


@pytest.fixture
def rng():
    return check_random_state(42)


@pytest.fixture
def X(rng, n_trials):
    return rng.randn(n_trials, N_CHANNELS, N_TIMES)


class MockZEncoder:

    def __init__(self, X, D_hat, z_hat, n_atoms, n_channels, n_times_atom,
                 loss, loss_params):
        self.X = X
        self.D_hat = D_hat
        self.z_hat = z_hat
        self.n_atoms = n_atoms
        self.n_channels = n_channels
        self.n_times_atom = n_times_atom
        self.loss = loss
        self.loss_params = loss_params

        self.ztX = compute_ztX(self.z_hat, self.X)
        self.ztz = compute_ztz(self.z_hat, N_TIMES_ATOM)
        self.XtX = np.dot(X.ravel(), X.ravel())

    def get_z_hat(self):
        return self.z_hat

    def get_constants(self):

        return dict(ztX=self.ztX, ztz=self.ztz, XtX=self.XtX,
                    n_channels=self.n_channels)

    def compute_objective(self, D):
        if self.loss == 'l2':
            return compute_objective(D=D,
                                     constants=self.get_constants())

        return compute_X_and_objective_multi(
            self.X,
            self.get_z_hat(),
            D_hat=D,
            loss=self.loss,
            loss_params=self.loss_params,
            feasible_evaluation=False
        )


@pytest.fixture
def z_encoder_rank1(rng, shape, loss):

    from alphacsc.utils import construct_X_multi
    from alphacsc.update_d_multi import prox_uv
    z_hat = rng.normal(size=(N_TRIALS, N_ATOMS, N_TIMES - N_TIMES_ATOM + 1))
    uv0 = rng.normal(size=shape)
    uv0 = prox_uv(uv0)

    X = construct_X_multi(z_hat, D=uv0, n_channels=N_CHANNELS)

    return MockZEncoder(X, uv0, z_hat, N_ATOMS, N_CHANNELS, N_TIMES_ATOM,
                        loss, dict())
