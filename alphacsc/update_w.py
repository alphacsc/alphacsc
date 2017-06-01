"""Convolutional dictionary learning"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>

import numpy as np
from scipy.stats import levy_stable

from .utils import check_random_state


def estimate_phi_mh(X, Xhat, alpha, Phi, n_iter_mcmc, n_burnin_mcmc,
                    random_state, return_loglk=False, verbose=10):
    """Estimate the expectation of 1/phi by Metropolis-Hastings"""

    if n_iter_mcmc <= n_burnin_mcmc:
        raise ValueError('n_iter_mcmc must be greater than n_burnin_mcmc')

    n_trials, n_times = X.shape
    residual = (X - Xhat) ** 2

    tau = np.zeros((n_trials, n_times))
    rng = check_random_state(random_state)

    if return_loglk:
        loglk_all = np.zeros((n_iter_mcmc, 1))

    for i in range(n_iter_mcmc):
        scale = 2 * np.cos(np.pi * alpha / 4) ** (2 / alpha)
        Phi_p = levy_stable.rvs(alpha / 2, 1, loc=0, scale=scale,
                                size=(n_trials, n_times), random_state=rng)
        log_acc = 0.5 * np.log(Phi / Phi_p) + residual * (1 / Phi - 1 / Phi_p)
        log_u = np.log(rng.uniform(size=(n_trials, n_times)))
        ix = (log_acc > log_u)

        Phi[ix] = Phi_p[ix]

        if return_loglk:
            loglk = np.sum(-0.5 * np.log(Phi) - 0.5 * residual / Phi)
            loglk_all[i] = loglk
            if verbose > 5:
                print("Iter: %d\t loglk:%E\t NumAcc:%d" %
                      (i, loglk, np.sum(ix)))

        if (i >= n_burnin_mcmc):
            tau += 1 / Phi

    tau = tau / (n_iter_mcmc - n_burnin_mcmc)

    if return_loglk:
        return Phi, tau, loglk_all
    return Phi, tau
