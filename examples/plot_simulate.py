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

from functools import partial

import matplotlib.pyplot as plt

from alphacsc.simulate import simulate_data
from alphacsc import learn_d_z, learn_d_z_weighted
from alphacsc import update_d_block, construct_X, check_random_state

reg = 0.1
alpha = 1.8

fraction_corrupteds = [0.0, 0.1]

n_times_atom = 64  # L
n_times = 512  # T
n_atoms = 2  # K
n_trials = 100  # N

n_iter_global = 5
n_iter_optim = 20
n_iter_optim_init = 20
n_iter_mcmc = 10
n_burnin_mcmc = 5

n_iter = n_iter_optim_init + n_iter_global * n_iter_optim  # fair comparison

random_state = 60
fraction_corrupted = 0.1

print('fraction_corrupted: %0.2f' % fraction_corrupted)
n_corrupted_trials = int(fraction_corrupted * n_trials)

random_state_simulate = 1
X, ds_true, Z_true = simulate_data(n_trials, n_times, n_times_atom,
                                   n_atoms, random_state_simulate)
X_hat = construct_X(Z_true, ds_true)

rng = check_random_state(random_state_simulate)

# Add stationary noise:
X += 0.01 * rng.randn(*X.shape)
noise_level = 0.005

# add corrupted trials
if n_corrupted_trials > 0:
    idx_corrupted = rng.randint(0, n_trials,
                                size=n_corrupted_trials)
    X[idx_corrupted] += 0.1 * rng.randn(n_corrupted_trials, X.shape[1])

func = partial(update_d_block, projection='dual')

pobj, times, d_hat, Z_hat = learn_d_z(
    X, n_atoms, n_times_atom, func_d=func, reg=reg,
    n_iter=n_iter,
    solver_d_kwargs=dict(factr=100), random_state=random_state,
    n_jobs=1, solver_z='l_bfgs', verbose=1)
print('Vanilla CSC')

d_hat_mcem, z_hat_mcem, Tau = learn_d_z_weighted(
    X, n_atoms, n_times_atom, func_d=func, reg=reg, alpha=alpha,
    solver_d_kwargs=dict(factr=100), n_iter_global=n_iter_global,
    n_iter_optim=n_iter_optim,
    n_iter_mcmc=n_iter_mcmc, n_burnin_mcmc=n_burnin_mcmc,
    random_state=random_state, n_jobs=1, solver_z='l_bfgs',
    verbose=1)

plt.plot(d_hat.T, 'b', label='CSC')
plt.plot(d_hat_mcem.T, 'r', label=r'$\alpha$CSC')
plt.plot(ds_true.T, 'k--', label='True atoms')
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::2], labels[::2], loc='best')

plt.title('%d%% corrupted' % (fraction_corrupted * 100))
plt.show()
