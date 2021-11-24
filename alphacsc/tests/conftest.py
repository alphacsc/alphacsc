import pytest


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
