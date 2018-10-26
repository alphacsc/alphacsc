import pytest
import numpy as np

from alphacsc.utils import check_random_state
from alphacsc.online_dictionary_learning import OnlineCDL

@pytest.mark.parametrize('rank1', [True, False])
@pytest.mark.parametrize('alpha', [.2, .8])
def test_learn_d_z_multi(rank1, alpha):
    # smoke test for learn_d_z_multi
    n_trials, n_channels, n_times = 10, 3, 100
    n_times_atom, n_atoms = 6, 4

    rng = check_random_state(42)
    X = rng.randn(n_trials, n_channels, n_times)

    # The initial regularization is different for fit and partial_fit. It is
    # computed in batch mode for fit and with the first mini-batch in
    # partial_fit.
    params = dict(n_atoms=n_atoms, n_times_atom=n_times_atom, n_iter=n_trials, lmbd_max="fixed",
                  rank1=rank1, batch_size=1, batch_selection='cyclic',
                  alpha=alpha, random_state=12)

    cdl_fit = OnlineCDL(**params)
    cdl_partial = OnlineCDL(**params)

    cdl_fit.fit(X)
    for x in X:
        cdl_partial.partial_fit(x[None])

    assert np.allclose(cdl_fit._D_hat, cdl_partial._D_hat)