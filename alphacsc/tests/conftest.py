import pytest

import numpy as np

from alphacsc.update_d_multi import prox_uv, prox_d
from alphacsc.utils.convolution import construct_X_multi
from alphacsc.utils.validation import check_random_state
from alphacsc.utils.compute_constants import compute_ztz, compute_ztX
from alphacsc.loss_and_gradient import compute_objective

N_TRIALS, N_CHANNELS, N_TIMES = 2, 3, 100
N_TIMES_ATOM, N_ATOMS = 6, 4

parametrize_solver_and_constraint = pytest.mark.parametrize(
    'rank1, solver_d, uv_constraint',
    [
        (True, 'alternate_adaptive', 'separate'),
        (True, 'alternate', 'separate'),
        (True, 'joint', 'joint'),
        (True, 'joint', 'separate'),
        (True, 'fista', 'joint'),
        (True, 'fista', 'separate'),
        (False, 'auto', 'auto'),
        (False, 'fista', 'auto'),
    ]
)


@pytest.fixture
def rng():
    return check_random_state(42)


@pytest.fixture
def n_trials():
    return N_TRIALS


@pytest.fixture
def X(rng, n_trials):
    return rng.randn(n_trials, N_CHANNELS, N_TIMES)


@pytest.fixture
def D_hat(rank1, rng):
    if rank1:
        shape = (N_ATOMS, N_CHANNELS + N_TIMES_ATOM)
        d = rng.normal(size=shape)
        return prox_uv(d)
    else:
        shape = (N_ATOMS, N_CHANNELS, N_TIMES_ATOM)
        d = rng.normal(size=shape)
        return prox_d(d)


class MockZEncoder:

    def __init__(self, X, D_hat, z_hat, n_atoms, n_channels, n_times_atom):
        self.X = X
        self.D_hat = D_hat
        self.z_hat = z_hat
        self.n_atoms = n_atoms
        self.n_channels = n_channels
        self.n_times_atom = n_times_atom

        self.ztX = compute_ztX(self.z_hat, self.X)
        self.ztz = compute_ztz(self.z_hat, N_TIMES_ATOM)
        self.XtX = np.dot(X.ravel(), X.ravel())

    def get_z_hat(self):
        return self.z_hat

    def set_D(self, D):
        self.D_hat = D

    def get_constants(self):

        return dict(ztX=self.ztX, ztz=self.ztz, XtX=self.XtX,
                    n_channels=self.n_channels)

    def compute_objective(self, D):
        return compute_objective(D=D, constants=self.get_constants())


@pytest.fixture
def z_encoder(D_hat, rng):

    z_hat = rng.normal(size=(N_TRIALS, N_ATOMS, N_TIMES - N_TIMES_ATOM + 1))

    X = construct_X_multi(z_hat, D=D_hat, n_channels=N_CHANNELS)

    return MockZEncoder(X, D_hat, z_hat, N_ATOMS, N_CHANNELS, N_TIMES_ATOM)
