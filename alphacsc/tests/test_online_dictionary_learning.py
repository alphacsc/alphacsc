import pytest
import numpy as np

from alphacsc.online_dictionary_learning import OnlineCDL
from alphacsc.tests.conftest import N_TIMES_ATOM, N_ATOMS


@pytest.mark.parametrize('solver, rank1, n_trials', [
    ('lgcd', True, 10),
    ('lgcd', False, 10),
    #   ('dicodile', 'False', 1)
])
@pytest.mark.parametrize('alpha', [.2, .8])
def test_online_partial_fit(X, solver, rank1, n_trials, alpha):

    X /= X.std()

    # The initial regularization is different for fit and partial_fit. It is
    # computed in batch mode for fit and with the first mini-batch in
    # partial_fit.
    params = dict(solver_z=solver, n_atoms=N_ATOMS, n_times_atom=N_TIMES_ATOM,
                  n_iter=n_trials,
                  D_init="random", lmbd_max="fixed", rank1=rank1, batch_size=1,
                  batch_selection='cyclic', alpha=alpha, random_state=12)

    cdl_fit = OnlineCDL(**params)
    cdl_partial = OnlineCDL(**params)

    cdl_fit.fit(X)
    for x in X:
        cdl_partial.partial_fit(x[None])

    assert np.allclose(cdl_fit._D_hat, cdl_partial._D_hat)
