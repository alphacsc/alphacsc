import pytest
import numpy as np

from alphacsc.utils.validation import check_random_state
from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.convolutional_dictionary_learning import BatchCDL, GreedyCDL
from alphacsc.online_dictionary_learning import OnlineCDL
from alphacsc.init_dict import init_dictionary

from alphacsc.tests.conftest import parametrize_solver_and_constraint

from .conftest import N_ATOMS, N_TIMES, N_TIMES_ATOM, N_CHANNELS


@pytest.mark.parametrize('window', [False, True])
@pytest.mark.parametrize('rank1, solver_d, uv_constraint',
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
@pytest.mark.parametrize('lmbd_max', ['fixed', 'scaled', 'shared', 'per_atom'])
def test_learn_d_z_multi(X, rank1, solver_d, uv_constraint, window,
                         lmbd_max):
    # smoke test for learn_d_z_multi

    pobj, times, uv_hat, z_hat, reg = learn_d_z_multi(
        X, N_ATOMS, N_TIMES_ATOM, uv_constraint=uv_constraint, rank1=rank1,
        solver_d=solver_d, random_state=0, lmbd_max=lmbd_max,
        n_iter=30, eps=-np.inf, solver_z='l-bfgs', window=window,
        verbose=0
    )

    msg = "Cost function does not go down for uv_constraint {}".format(
        uv_constraint)

    try:
        if lmbd_max != 'shared' and lmbd_max != 'per_atom':
            assert np.sum(np.diff(pobj) > 1e-13) == 0, msg
    except AssertionError:
        import matplotlib.pyplot as plt
        plt.semilogy(pobj - np.min(pobj) + 1e-6)
        plt.title(msg)
        plt.show()
        raise


@pytest.mark.parametrize('window', [False, True])
@pytest.mark.parametrize('rank1', [False, True])
@pytest.mark.parametrize('n_trials', [1])
def test_learn_d_z_multi_dicodile(X, window, rank1):
    pytest.importorskip('dicodile')
    # smoke test for learn_d_z_multi

    pobj, times, uv_hat, z_hat, reg = learn_d_z_multi(
        X, N_ATOMS, N_TIMES_ATOM, uv_constraint='auto', rank1=rank1,
        solver_d='auto', random_state=0,
        n_iter=30, eps=-np.inf, solver_z='dicodile', window=window,
        verbose=0
    )

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
def test_window(X, rank1, solver_d, uv_constraint):
    # Smoke test that the parameter window does something

    D_init = init_dictionary(X, N_ATOMS, N_TIMES_ATOM, rank1=rank1,
                             uv_constraint=uv_constraint, random_state=0,
                             window=False)

    kwargs = dict(X=X, n_atoms=N_ATOMS, n_times_atom=N_TIMES_ATOM, verbose=0,
                  uv_constraint=uv_constraint, solver_d=solver_d, rank1=rank1,
                  random_state=0, n_iter=1, solver_z='l-bfgs', D_init=D_init)
    res_False = learn_d_z_multi(window=False, **kwargs)

    D_init = init_dictionary(X, N_ATOMS, N_TIMES_ATOM, rank1=rank1,
                             uv_constraint=uv_constraint, random_state=0,
                             window=True)

    kwargs = dict(X=X, n_atoms=N_ATOMS, n_times_atom=N_TIMES_ATOM, verbose=0,
                  uv_constraint=uv_constraint, solver_d=solver_d, rank1=rank1,
                  random_state=0, n_iter=1, solver_z='l-bfgs', D_init=D_init)

    res_True = learn_d_z_multi(window=True, **kwargs)

    assert not np.allclose(res_False[2], res_True[2])


def test_online_learning(X, n_trials):
    # smoke test for learn_d_z_multi

    pobj_0, _, _, _, _ = learn_d_z_multi(
        X, N_ATOMS, N_TIMES_ATOM, uv_constraint="separate", solver_d="joint",
        random_state=0, n_iter=30, solver_z='l-bfgs', algorithm="batch",
    )

    pobj_1, _, _, _, _ = learn_d_z_multi(
        X, N_ATOMS, N_TIMES_ATOM, uv_constraint="separate", solver_d="joint",
        random_state=0, n_iter=30, solver_z='l-bfgs', algorithm="online",
        algorithm_params=dict(batch_size=n_trials, alpha=0)
    )

    assert np.allclose(pobj_0, pobj_1)


@pytest.mark.parametrize('klass', [BatchCDL, OnlineCDL, GreedyCDL])
@parametrize_solver_and_constraint
def test_transformers(X, klass, rank1, solver_d, uv_constraint, n_trials):
    # smoke test for transformer classes

    if klass == OnlineCDL:
        kwargs = dict(batch_selection='cyclic')
    else:
        kwargs = dict()

    rng = check_random_state(42)
    X = rng.randn(n_trials, N_CHANNELS, N_TIMES)
    cdl = klass(N_ATOMS, N_TIMES_ATOM, uv_constraint=uv_constraint,
                rank1=rank1, solver_d=solver_d, random_state=0, n_iter=10,
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
def test_unbiased_z_hat(X, solver_z):

    _, _, _, z_hat, _ = learn_d_z_multi(
        X, N_ATOMS, N_TIMES_ATOM, uv_constraint='auto', rank1=False,
        solver_d='auto', random_state=0, unbiased_z_hat=False,
        n_iter=1, eps=-np.inf, solver_z=solver_z, window=False,
        verbose=0)

    _, _, _, z_hat_unbiased, _ = learn_d_z_multi(
        X, N_ATOMS, N_TIMES_ATOM, uv_constraint='auto', rank1=False,
        solver_d='auto', random_state=0, unbiased_z_hat=True,
        n_iter=1, eps=-np.inf, solver_z=solver_z, window=False,
        verbose=0)

    assert np.all(z_hat_unbiased[z_hat == 0] == 0)


@pytest.mark.parametrize('rank1', [False, True])
@pytest.mark.parametrize('n_iter', [0, 1, 5])
def test_learn_d_z_multi_solver_z(rank1, n_iter):
    pytest.importorskip('dicodile')

    X = np.random.randn(1, 20, 256)

    n_atoms = 2
    n_times_atom = 5
    z_max_iter = 1
    z_tol = 1e-3
    eps = 1e-3
    reg = 0.1
    njobs = 1
    ds_init = "chunk"
    random_state = 0
    solver_z_kwargs = dict(max_iter=z_max_iter, tol=z_tol)

    pobj_g, _, d_hat_g, z_hat_g, reg_g = learn_d_z_multi(
        X, n_atoms, n_times_atom, solver_d='auto', solver_z="lgcd",
        uv_constraint='auto', eps=eps, solver_z_kwargs=solver_z_kwargs,
        reg=reg, solver_d_kwargs=dict(max_iter=100), n_iter=n_iter,
        random_state=random_state, raise_on_increase=False, D_init=ds_init,
        n_jobs=njobs, rank1=rank1)

    pobj_d, _, d_hat_d, z_hat_d, reg_d = learn_d_z_multi(
        X, n_atoms, n_times_atom, solver_d='auto', solver_z="dicodile",
        uv_constraint='auto', eps=eps, solver_z_kwargs=solver_z_kwargs,
        reg=reg, solver_d_kwargs=dict(max_iter=100), n_iter=n_iter,
        random_state=random_state, raise_on_increase=False, D_init=ds_init,
        n_jobs=njobs, rank1=rank1)

    assert reg_d == reg_g
    assert np.allclose(d_hat_d, d_hat_g, rtol=1e-9, atol=1e-9)
    assert np.allclose(z_hat_d, z_hat_g, rtol=1e-9, atol=1e-9)
    assert np.allclose(pobj_d, pobj_g, rtol=1e-9, atol=1e-9)
