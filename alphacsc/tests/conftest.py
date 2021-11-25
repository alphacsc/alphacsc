import pytest

from alphacsc.utils import check_random_state

N_TRIALS, N_CHANNELS, N_TIMES = 5, 3, 100

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
