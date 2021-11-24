import pytest

from alphacsc._solver_d import check_solver_and_constraints, get_solver_d

from alphacsc.tests.conftest import parametrize_solver_and_constraint


@pytest.mark.parametrize('solver_d', ['auto', 'fista'])
@pytest.mark.parametrize('uv_constraint', ['auto'])
def test_check_solver_and_constraints(solver_d, uv_constraint):
    """Tests for the case rank1 is False."""

    solver_d_, uv_constraint_ = check_solver_and_constraints(False, solver_d,
                                                             uv_constraint)

    assert solver_d_ == 'fista'
    assert uv_constraint_ == 'auto'


@pytest.mark.parametrize('solver_d', ['auto', 'fista'])
@pytest.mark.parametrize('uv_constraint', ['joint', 'separate'])
def test_check_solver_and_constraints_error(solver_d, uv_constraint):
    """Tests for the case rank1 is False and params are not compatible."""

    with pytest.raises(AssertionError,
                       match="If rank1 is False, uv_constraint should be*"):

        check_solver_and_constraints(False, solver_d, uv_constraint)


@pytest.mark.parametrize('solver_d', ['auto', 'alternate',
                                      'alternate_adaptive'])
@pytest.mark.parametrize('uv_constraint', ['auto', 'separate'])
def test_check_solver_and_constraints_rank1(solver_d, uv_constraint):
    """Tests for the case rank1 is True."""

    solver_d_, uv_constraint_ = check_solver_and_constraints(True, solver_d,
                                                             uv_constraint)

    if solver_d == 'auto':
        solver_d = 'alternate_adaptive'

    assert solver_d_ == solver_d
    assert uv_constraint_ == 'separate'


@pytest.mark.parametrize('solver_d', ['joint', 'fista'])
@pytest.mark.parametrize('uv_constraint', ['auto', 'joint', 'separate'])
def test_check_solver_and_constraints_rank1_(solver_d, uv_constraint):
    """Tests for the case rank1 is True."""

    solver_d_, uv_constraint_ = check_solver_and_constraints(True, solver_d,
                                                             uv_constraint)

    if uv_constraint == 'auto':
        uv_constraint = 'joint'

    assert solver_d_ == solver_d
    assert uv_constraint_ == uv_constraint


@pytest.mark.parametrize('solver_d', ['auto', 'alternate',
                                      'alternate_adaptive'])
@pytest.mark.parametrize('uv_constraint', ['joint'])
def test_check_solver_and_constraints_rank1_error(solver_d, uv_constraint):
    """Tests for error the case when rank1 is True and params are not compatible.
    """
    with pytest.raises(AssertionError,
                       match="solver_d=*"):

        check_solver_and_constraints(True, solver_d, uv_constraint)


@parametrize_solver_and_constraint
@pytest.mark.parametrize('window', [True, False])
@pytest.mark.parametrize('momentum', [True, False])
def test_get_solver_d(rank1, solver_d, uv_constraint, window, momentum):
    """Tests valid values."""

    d_solver = get_solver_d(solver_d=solver_d,
                            rank1=rank1,
                            uv_constraint=uv_constraint,
                            window=window,
                            momentum=momentum)

    assert d_solver is not None


@pytest.mark.parametrize('solver_d', ['auto', 'fista'])
@pytest.mark.parametrize('uv_constraint', ['joint', 'separate'])
@pytest.mark.parametrize('window', [True, False])
@pytest.mark.parametrize('momentum', [True, False])
def test_get_solver_d_error(solver_d, uv_constraint, window, momentum):
    """Tests for the case rank1 is False and params are not compatible."""

    with pytest.raises(AssertionError,
                       match="If rank1 is False, uv_constraint should be*"):

        get_solver_d(solver_d=solver_d,
                     rank1=False,
                     uv_constraint=uv_constraint,
                     window=window,
                     momentum=momentum)


@pytest.mark.parametrize('solver_d', ['auto', 'alternate',
                                      'alternate_adaptive'])
@pytest.mark.parametrize('uv_constraint', ['joint'])
@pytest.mark.parametrize('window', [True, False])
@pytest.mark.parametrize('momentum', [True, False])
def test_get_solver_d_rank1_error(solver_d, uv_constraint, window, momentum):
    """Tests for error the case when rank1 is True and params are not compatible.
    """
    with pytest.raises(AssertionError,
                       match="solver_d=*"):

        get_solver_d(solver_d=solver_d,
                     rank1=False,
                     uv_constraint=uv_constraint,
                     window=window,
                     momentum=momentum)
