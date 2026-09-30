# -*- coding: utf-8 -*-
"""
==================
No-overlap example
==================

In this example, we use the no-overlap solver on an open dataset of gait (steps)
IMU time-series to discover patterns in the data. We will then use those to
attempt to detect steps and compare our findings with the ground truth.
"""

###############################################################################
# Using 'no-overlap'
# ------------------
#
# Retrieve trial data

from dicodile.data.gait import get_gait_data

n_trials = 5
channels = ['RAV', 'RRY', 'RAX']

trials = []
for i in range(n_trials):
    trial = get_gait_data(subject=6, trial=i+1)
    trials.append(trial["data"][channels].T)

###############################################################################
# We now store the trials into a unique numpy array X

import numpy as np

n_times = max(x.shape[1] for x in trials)
X = np.zeros((n_trials, len(channels), n_times), dtype=np.float64)

for i in range(n_trials):
    xtx = np.linalg.norm(trials[i], axis=1)
    X[i,:,:trials[i].shape[-1]] = trials[i] / xtx[:,None]


###############################################################################
# We then set the parameters of the dictionary learning

# set dictionary size
n_atoms = 12

# set individual atom (patch) size.
n_times_atom = 120

# set regularization parameter
lmbd = 0.1 * n_times_atom / n_times

###############################################################################
# We are now able to intialize a dictionary using a strategy for the
# non-overlapping case

from alphacsc.init_dict import init_dictionary

D0 = init_dictionary(
    X,
    n_atoms=n_atoms, n_times_atom=n_times_atom,
    rank1=False, D_init='no-overlap', reg=lmbd
)


###############################################################################
# The dictionary can be optimized using l0 normalization and assuming no
# overlap between atoms in the signal encoding as follow

from alphacsc import BatchCDL

cdl = BatchCDL(
    n_atoms, n_times_atom, 
    D_init=D0, rank1=False, 
    solver_z="no-overlap", solver_d="no-overlap",
    lmbd_max="fixed", reg=lmbd / X.var(), n_iter=5
)

cdl.fit(X)


###############################################################################
# Check that our representation is indeed sparse:

nnz = np.count_nonzero(cdl._z_hat)
print("Non zero coefficients count:", nnz)

###############################################################################
# Now, let's reconstruct the original signal.

from alphacsc.utils.convolution import construct_X_multi

def reconstruct(cdl):
    X_hat = construct_X_multi(cdl._z_hat, cdl._D_hat)
    diff = X - X_hat
    E0 = .5 * diff.ravel() @ diff.ravel() / X.var()
    E = E0 + lmbd * nnz / X.var()
    print("l2-objective:", round(E0, 1))
    print("BatchCDL 'no-overlap' objective:", round(E, 1))
    return X_hat

X_hat = reconstruct(cdl)

###############################################################################
# Plot a small part of the original and reconstructed signals

import matplotlib.pyplot as plt

trial = 0
channel = 0

fig_hat, ax_hat = plt.subplots()
ax_hat.plot(X[trial, channel, 5000:5800],
            label='right foot vertical acceleration (ORIGINAL)')
ax_hat.plot(X_hat[trial, channel, 5000:5800],
            label='right foot vertical acceleration (RECONSTRUCTED)')
ax_hat.set_xlabel('time (x10ms)')
ax_hat.set_ylabel('acceleration ($m.s^{-2}$)')
ax_hat.legend()
plt.show()

###############################################################################
# Plot the atoms learned inside the dictionary

from dicodile.utils.viz import display_dictionaries

display_dictionaries(cdl._D_hat)

###############################################################################
# Comparisons
# -----------
#
# Let's compare timings and objectives when using different solvers and
# initializers

import time

lgcd_reg = 4. * lmbd * n_times / n_times_atom / X.std()

args = [
    {"solver_z": 'no-overlap', "D_init": 'no-overlap', "solver_d": 'no-overlap'},
    {"solver_z": 'no-overlap', "D_init": 'no-overlap', "solver_d": 'fista'},
    {"solver_z": 'no-overlap', "D_init": 'random', "solver_d": 'no-overlap'},
    {"solver_z": 'no-overlap', "D_init": 'random', "solver_d": 'fista'},
    {"solver_z": 'lgcd', "D_init": 'no-overlap', "solver_d": 'fista'},
    {"solver_z": 'lgcd', "D_init": 'random', "solver_d": 'fista'},
]

for kwarg in args:
    print(kwarg)
    print("="*len(repr(kwarg)))

    reg = lgcd_reg if kwarg["solver_z"] == 'lgcd' else lmbd / X.var()
    if kwarg["D_init"] == 'no-overlap':
        kwarg['init_kwargs'] = {"reg": reg}
    cdl2 = BatchCDL(
        n_atoms, n_times_atom, 
        random_state=42, rank1=False, 
        lmbd_max="fixed", reg=reg, n_iter=5,
        verbose=0, **kwarg
    )
    start = time.time()
    cdl2.fit(X)
    fit_time = time.time() - start

    print("time:", fit_time)
    reconstruct(cdl2)
    print()
