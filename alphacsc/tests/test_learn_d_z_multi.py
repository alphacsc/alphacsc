from alphacsc.update_d_multi import check_solver_and_constraints
import pytest
import numpy as np

from alphacsc.utils import check_random_state
from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.convolutional_dictionary_learning import BatchCDL, GreedyCDL
from alphacsc.online_dictionary_learning import OnlineCDL
from alphacsc.init_dict import init_dictionary

from alphacsc.tests.conftest import parametrize_solver_and_constraint


@pytest.mark.parametrize('window', [False, True])
@pytest.mark.parametrize('loss', ['l2', 'dtw', 'whitening'])
@parametrize_solver_and_constraint
def test_learn_d_z_multi(loss, rank1, solver_d, uv_constraint, window):
    # smoke test for learn_d_z_multi
    n_trials, n_channels, n_times = 2, 3, 30
    n_times_atom, n_atoms = 6, 4

    loss_params = dict(gamma=1, sakoe_chiba_band=10, ordar=10)

    rng = check_random_state(42)
    X = rng.randn(n_trials, n_channels, n_times)
    pobj, times, uv_hat, z_hat, reg = learn_d_z_multi(
        X, n_atoms, n_times_atom, uv_constraint=uv_constraint, rank1=rank1,
        solver_d=solver_d, random_state=0,
        n_iter=30, eps=-np.inf, solver_z='l-bfgs', window=window,
        verbose=0, loss=loss, loss_params=loss_params)

    msg = "Cost function does not go down for uv_constraint {}".format(
        uv_constraint)

    try:
        assert np.sum(np.diff(pobj) > 1e-13) == 0, msg
    except AssertionError:
        import matplotlib.pyplot as plt
        plt.semilogy(pobj - np.min(pobj) + 1e-6)
        plt.title(msg)
        plt.show()
        raise


@pytest.mark.parametrize('window', [False, True])
@pytest.mark.parametrize('loss', ['dtw', 'whitening'])
@pytest.mark.parametrize('rank1, solver_d, uv_constraint', [
    (True, 'alternate', 'separate')])
def test_learn_d_z_multi_error(loss, rank1, solver_d, uv_constraint, window):
    # smoke test for learn_d_z_multi
    n_trials, n_channels, n_times = 2, 3, 30
    n_times_atom, n_atoms = 6, 4

    loss_params = dict(gamma=1, sakoe_chiba_band=10, ordar=10)

    rng = check_random_state(42)
    X = rng.randn(n_trials, n_channels, n_times)

    with pytest.raises(NotImplementedError):
        pobj, times, uv_hat, z_hat, reg = learn_d_z_multi(
            X, n_atoms, n_times_atom, uv_constraint=uv_constraint, rank1=rank1,
            solver_d=solver_d, random_state=0,
            n_iter=30, eps=-np.inf, solver_z='l-bfgs', window=window,
            verbose=0, loss=loss, loss_params=loss_params
        )


@pytest.mark.parametrize('window', [False, True])
# TODO expand params when DiCoDiLe is rank-1-capable
def test_learn_d_z_multi_dicodile(window):
    pytest.importorskip('dicodile')
    # smoke test for learn_d_z_multi
    # XXX For DiCoDiLe, n_trials cannot be >1
    n_trials, n_channels, n_times = 1, 3, 30
    n_times_atom, n_atoms = 6, 4

    rng = check_random_state(42)
    X = rng.randn(n_trials, n_channels, n_times)
    pobj, times, uv_hat, z_hat, reg = learn_d_z_multi(
        X, n_atoms, n_times_atom, uv_constraint='auto', rank1=False,
        solver_d='auto', random_state=0,
        n_iter=30, eps=-np.inf, solver_z='dicodile', window=window,
        verbose=0, loss='l2', loss_params=None)

    msg = "Cost function does not go down"

    try:
        assert np.sum(np.diff(pobj) > 1e-13) == 0, msg
    except AssertionError:
        import matplotlib.pyplot as plt
        plt.semilogy(pobj - np.min(pobj) + 1e-6)
        plt.title(msg)
        plt.show()
        raise


@parametrize_solver_and_constraint
def test_window(rank1, solver_d, uv_constraint):
    # Smoke test that the parameter window does something
    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4

    rng = check_random_state(42)
    X = rng.randn(n_trials, n_channels, n_times)

    *_, uv_constraint_ = check_solver_and_constraints(
        rank1, solver_d, uv_constraint
    )

    D_init = init_dictionary(X, n_atoms, n_times_atom, rank1=rank1,
                             uv_constraint=uv_constraint_, random_state=0,
                             window=False)

    kwargs = dict(X=X, n_atoms=n_atoms, n_times_atom=n_times_atom, verbose=0,
                  uv_constraint=uv_constraint, solver_d=solver_d, rank1=rank1,
                  random_state=0, n_iter=1, solver_z='l-bfgs', D_init=D_init)
    res_False = learn_d_z_multi(window=False, **kwargs)

    D_init = init_dictionary(X, n_atoms, n_times_atom, rank1=rank1,
                             uv_constraint=uv_constraint_, random_state=0,
                             window=True)

    kwargs = dict(X=X, n_atoms=n_atoms, n_times_atom=n_times_atom, verbose=0,
                  uv_constraint=uv_constraint, solver_d=solver_d, rank1=rank1,
                  random_state=0, n_iter=1, solver_z='l-bfgs', D_init=D_init)
    res_True = learn_d_z_multi(window=True, **kwargs)

    assert not np.allclose(res_False[2], res_True[2])


def test_online_learning():
    # smoke test for learn_d_z_multi
    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4

    rng = check_random_state(42)
    X = rng.randn(n_trials, n_channels, n_times)
    pobj_0, _, _, _, _ = learn_d_z_multi(
        X, n_atoms, n_times_atom, uv_constraint="separate", solver_d="joint",
        random_state=0, n_iter=30, solver_z='l-bfgs', algorithm="batch",
        loss='l2')

    pobj_1, _, _, _, _ = learn_d_z_multi(
        X, n_atoms, n_times_atom, uv_constraint="separate", solver_d="joint",
        random_state=0, n_iter=30, solver_z='l-bfgs', algorithm="online",
        algorithm_params=dict(batch_size=n_trials, alpha=0), loss='l2')

    assert np.allclose(pobj_0, pobj_1)


@pytest.mark.parametrize('klass', [BatchCDL, OnlineCDL, GreedyCDL])
def test_transformers(klass):
    # smoke test for transformer classes
    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4

    if klass == OnlineCDL:
        kwargs = dict(batch_selection='cyclic')
    else:
        kwargs = dict()

    rng = check_random_state(42)
    X = rng.randn(n_trials, n_channels, n_times)
    cdl = klass(n_atoms, n_times_atom, uv_constraint='separate', rank1=True,
                solver_d='alternate_adaptive', random_state=0, n_iter=10,
                eps=-np.inf, solver_z='l-bfgs', window=True, verbose=0,
                **kwargs)
    cdl.fit(X)
    z = cdl.transform(X)
    Xt = cdl.transform_inverse(z)
    assert Xt.shape == X.shape

    msg = "Cost function does not go down for %s" % klass
    assert np.sum(np.diff(cdl.pobj_) > 1e-13) == 0, msg

    attributes = [
        'D_hat_', 'uv_hat_', 'u_hat_', 'v_hat_', 'z_hat_', 'pobj_', 'times_'
    ]
    for attribute in attributes:
        getattr(cdl, attribute)


@pytest.mark.parametrize('solver_z', ['l-bfgs', 'lgcd'])
def test_unbiased_z_hat(solver_z):
    n_trials, n_channels, n_times = 2, 3, 30
    n_times_atom, n_atoms = 6, 4

    loss_params = dict(gamma=1, sakoe_chiba_band=10, ordar=10)

    rng = check_random_state(42)
    X = rng.randn(n_trials, n_channels, n_times)

    _, _, _, z_hat, _ = learn_d_z_multi(
        X, n_atoms, n_times_atom, uv_constraint='auto', rank1=False,
        solver_d='auto', random_state=0, unbiased_z_hat=False,
        n_iter=1, eps=-np.inf, solver_z=solver_z, window=False,
        verbose=0, loss='l2', loss_params=loss_params)

    _, _, _, z_hat_unbiased, _ = learn_d_z_multi(
        X, n_atoms, n_times_atom, uv_constraint='auto', rank1=False,
        solver_d='auto', random_state=0, unbiased_z_hat=True,
        n_iter=1, eps=-np.inf, solver_z=solver_z, window=False,
        verbose=0, loss='l2', loss_params=loss_params)

    assert np.all(z_hat_unbiased[z_hat == 0] == 0)
