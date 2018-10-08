import pytest
import numpy as np

from alphacsc.utils import check_random_state
from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.convolutional_dictionary_learning import BatchCDL, OnlineCDL
from alphacsc.init_dict import init_dictionary


@pytest.mark.parametrize('window', [False, True])
@pytest.mark.parametrize('loss', ['l2', 'dtw', 'whitening'])
@pytest.mark.parametrize(
    'solver_d, uv_constraint, rank1',
    [
        ('joint', 'joint', True),
        ('joint', 'separate', True),
        ('joint', 'joint', False),  # ('alternate', 'separate', True),
        ('alternate_adaptive', 'separate', True)
    ])
def test_learn_d_z_multi(loss, solver_d, uv_constraint, rank1, window):
    # smoke test for learn_d_z_multi
    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4

    loss_params = dict(gamma=1, sakoe_chiba_band=10, ordar=10)

    rng = check_random_state(42)
    X = rng.randn(n_trials, n_channels, n_times)
    pobj, times, uv_hat, z_hat, reg = learn_d_z_multi(
        X, n_atoms, n_times_atom, uv_constraint=uv_constraint, rank1=rank1,
        solver_d=solver_d, random_state=0, n_iter=30, eps=-np.inf,
        solver_z='l-bfgs', window=window, verbose=0, loss=loss,
        loss_params=loss_params)

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


@pytest.mark.parametrize('solver_d, uv_constraint, rank1',
                         [('joint', 'joint', True), ('joint', 'separate',
                                                     True),
                          ('joint', 'joint', False), ('alternate_adaptive',
                                                      'separate', True)])
def test_window(solver_d, uv_constraint, rank1):
    # Smoke test that the parameter window does something
    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4

    rng = check_random_state(42)
    X = rng.randn(n_trials, n_channels, n_times)

    D_init = init_dictionary(X, n_atoms, n_times_atom, rank1=True,
                             uv_constraint=uv_constraint, random_state=0)

    kwargs = dict(X=X, n_atoms=n_atoms, n_times_atom=n_times_atom, verbose=0,
                  uv_constraint=uv_constraint, solver_d=solver_d,
                  random_state=0, n_iter=1, solver_z='l-bfgs', D_init=D_init)
    res_False = learn_d_z_multi(window=False, **kwargs)
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


@pytest.mark.parametrize('klass', [BatchCDL, OnlineCDL])
def test_transformers(klass):
    # smoke test for transformer classes
    n_trials, n_channels, n_times = 2, 3, 100
    n_times_atom, n_atoms = 10, 4

    rng = check_random_state(42)
    X = rng.randn(n_trials, n_channels, n_times)
    cdl = klass(n_atoms, n_times_atom, uv_constraint='separate', rank1=True,
                solver_d='alternate_adaptive', random_state=0, n_iter=10,
                eps=-np.inf, solver_z='l-bfgs', window=True, verbose=0)
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
