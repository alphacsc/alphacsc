"""
===========================
Alpha CSC on simulated data
===========================
This example demonstrates alphaCSC on simulated data.

"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

###############################################################################
# Let us first define the parameters of our model.

n_times_atom = 64  # L
n_times = 512  # T
n_atoms = 2  # K
n_trials = 100  # N

alpha = 1.2

###############################################################################
# Next, we define the parameters for alpha CSC

n_iter_global = 10
n_iter_optim = 20
n_iter_mcmc = 100
n_burnin_mcmc = 50

n_iter = n_iter_global * n_iter_optim  # fair comparison

###############################################################################
# Here, we simulate the data

from alphacsc.simulate import simulate_data # noqa

random_state_simulate = 1
X, ds_true, Z_true = simulate_data(n_trials, n_times, n_times_atom,
                                   n_atoms, random_state_simulate)

###############################################################################
# Add some noise and corrupt some trials

from scipy.stats import levy_stable # noqa
from alphacsc import check_random_state # noqa

fraction_corrupted = 0.01
print('fraction_corrupted: %0.2f' % fraction_corrupted)
n_corrupted_trials = int(fraction_corrupted * n_trials)

# Add stationary noise:
rng = check_random_state(random_state_simulate)
X += 0.01 * rng.randn(*X.shape)

noise_level = 0.005
# add impulsive noise
if n_corrupted_trials > 0:
    idx_corrupted = rng.randint(0, n_trials,
                                size=n_corrupted_trials)
    X[idx_corrupted] += levy_stable.rvs(alpha, 0, loc=0, scale=noise_level,
                                        size=(n_corrupted_trials, n_times),
                                        random_state=random_state_simulate)

###############################################################################
# Now, we run vanilla CSC on the data.

from functools import partial # noqa
from alphacsc import learn_d_z, update_d_block # noqa

reg = 0.2  # twice the regularization in alpha-CSC
random_state = 60
func = partial(update_d_block, projection='dual')

pobj, times, d_hat, Z_hat = learn_d_z(
    X, n_atoms, n_times_atom, func_d=func, reg=reg,
    n_iter=n_iter,
    solver_d_kwargs=dict(factr=100), random_state=random_state,
    n_jobs=1, solver_z='l_bfgs', verbose=1)
print('Vanilla CSC')

###############################################################################
# and then alpha CSC on the same data

from alphacsc import learn_d_z_weighted # noqa

reg = 0.1  # let's give CSC the benefit of doubt :)
d_hat_mcem, z_hat_mcem, Tau = learn_d_z_weighted(
    X, n_atoms, n_times_atom, func_d=func, reg=reg, alpha=alpha,
    solver_d_kwargs=dict(factr=100), n_iter_global=n_iter_global,
    n_iter_optim=n_iter_optim,
    n_iter_mcmc=n_iter_mcmc, n_burnin_mcmc=n_burnin_mcmc,
    random_state=random_state, n_jobs=1, solver_z='l_bfgs',
    verbose=1)

###############################################################################
# Finally, let's compare the results.

import matplotlib.pyplot as plt # noqa

plt.plot(d_hat.T, 'b', label='CSC')
plt.plot(d_hat_mcem.T, 'r', label=r'$\alpha$CSC')
plt.plot(ds_true.T, 'k--', label='True atoms')
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::2], labels[::2], loc='best')
plt.show()
