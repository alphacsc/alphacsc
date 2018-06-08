"""Convolutional dictionary learning"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>

import numpy as np

from .utils import construct_X, check_random_state
from .learn_d_z import learn_d_z
from .update_d import update_d_block
from .update_w import estimate_phi_mh


def learn_d_z_weighted(X, n_atoms, n_times_atom, func_d=update_d_block,
                       reg=0.1, alpha=1.9,
                       n_iter_global=10, init_tau=False, n_iter_optim=10,
                       n_iter_mcmc=10, n_burnin_mcmc=0, random_state=None,
                       n_jobs=1, solver_z='l_bfgs', solver_d_kwargs=dict(),
                       solver_z_kwargs=dict(), verbose=10, callback=None):
    """Learn atoms using alphaCSC."""

    n_trials, n_times = X.shape

    if init_tau:
        Phi = np.tile(np.std(X, axis=1)[:, None] ** 2, X.shape[1])
        Tau = 1 / Phi
    else:
        # assume gaussian to start with
        Phi = 2 * np.ones((n_trials, n_times))
        Tau = 0.5 * np.ones((n_trials, n_times))

    rng = check_random_state(random_state)

    d_hat = None
    # Run the MCEM algorithm
    for i in range(n_iter_global):

        Tau *= 2

        # Optimize d and z wrt the new weights
        pobj, times, d_hat, z_hat = learn_d_z(
            X, n_atoms, n_times_atom, func_d, reg=reg, n_iter=n_iter_optim,
            random_state=rng, sample_weights=Tau, ds_init=d_hat,
            solver_d_kwargs=solver_d_kwargs, solver_z_kwargs=solver_z_kwargs,
            verbose=verbose, solver_z=solver_z, n_jobs=n_jobs,
            callback=callback)

        # Estimate the expectation via MCMC
        X_hat = construct_X(z_hat, d_hat)
        Phi, Tau, loglk_mcmc = estimate_phi_mh(
            X, X_hat, alpha, Phi, n_iter_mcmc, n_burnin_mcmc, random_state=rng,
            return_loglk=True, verbose=verbose)

        if verbose > 0:
            print("Global Iter: %d\t" % i)

    return d_hat, z_hat, Tau
