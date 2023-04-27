# -*- coding: utf-8 -*-
"""
====================
Gait (steps) example
====================
In this example, we use DiCoDiLe on an open dataset of gait (steps) IMU
time-series to discover patterns in the data. We will then use those to attempt
to detect steps and compare our findings with the ground truth.
"""

import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# # Retrieve trial data

from dicodile.data.gait import get_gait_data

trial = get_gait_data(subject=6, trial=1)

###############################################################################
# Let’s have a look at the data for one trial.

trial.keys()

###############################################################################
# We get a dictionary whose keys are metadata items, plus a ‘data’ key that
# contains a numpy array with the trial time series for each sensor axis, at
# 100 Hz resolution.

# right foot acceleration (vertical)
plt.plot(trial['data']['RAV'])

###############################################################################
# Let’s look at a small portion of the series for both feet, overlaid on the
# same plot.

fig, ax = plt.subplots()
ax.plot(trial['data']['LAV'][5000:5800],
        label='left foot vertical acceleration')
ax.plot(trial['data']['RAV'][5000:5800],
        label='right foot vertical acceleration')
ax.set_xlabel('time (x10ms)')
ax.set_ylabel('acceleration ($m.s^{-2}$)')
ax.legend()

###############################################################################
# We can see the alternating left and right foot movements.
#
# In the rest of this example, we will only use the right foot vertical
# acceleration.

###############################################################################
# # Convolutional Dictionary Learning

###############################################################################
# Now, let’s use "dicodile" as solver_z to learn patterns from the data and
# reconstruct the signal from a sparse representation.
#
# First, we initialize a dictionary from parts of the signal:

X = trial['data']['RAV'].to_numpy()

# reshape X to (n_trials, n_channels, n_times)
X = X.reshape(1, 1, *X.shape)

print(X.shape)

###############################################################################
# Note the use of reshape to shape the signal as per alphacsc requirements: the
# shape of the signal should be (n_trials, n_channels, n_times). Here, we have
# a single-channel time series so it is (1, 1, n_times).

from alphacsc.init_dict import init_dictionary

# set dictionary size
n_atoms = 8

# set individual atom (patch) size.
n_times_atom = 200

D_init = init_dictionary(X,
                         n_atoms=8,
                         n_times_atom=200,
                         rank1=False,
                         window=True,
                         D_init='chunk',
                         random_state=60)

print(D_init.shape)

""
from alphacsc import BatchCDL

cdl = BatchCDL(
    # Shape of the dictionary
    n_atoms,
    n_times_atom,
    rank1=False,
    uv_constraint='auto',
    # Number of iteration for the alternate minimization and cvg threshold
    n_iter=3,
    # number of workers to be used for dicodile
    n_jobs=4,
    # solver for the z-step
    solver_z='dicodile',
    solver_z_kwargs={'max_iter': 10000},
    window=True,
    D_init=D_init,
    random_state=60)

res = cdl.fit(X)

""
from dicodile.utils.viz import display_dictionaries

D_hat = res._D_hat

fig = display_dictionaries(D_init, D_hat)

###############################################################################
# # Signal reconstruction

###############################################################################
# Now, let's reconstruct the original signal.

from alphacsc.utils.convolution import construct_X_multi

z_hat = res._z_hat

X_hat = construct_X_multi(z_hat, D_hat)

###############################################################################
# Plot a small part of the original and reconstructed signals

fig_hat, ax_hat = plt.subplots()
ax_hat.plot(X[0][0][5000:5800],
            label='right foot vertical acceleration (ORIGINAL)')
ax_hat.plot(X_hat[0][0][5000:5800],
            label='right foot vertical acceleration (RECONSTRUCTED)')
ax_hat.set_xlabel('time (x10ms)')
ax_hat.set_ylabel('acceleration ($m.s^{-2}$)')
ax_hat.legend()

###############################################################################
# Check that our representation is indeed sparse:


np.count_nonzero(z_hat)

###############################################################################
# Besides our visual check, a measure of how closely we’re reconstructing the
# original signal is the (normalized) cross-correlation. Let’s compute this:

np.correlate(X[0][0], X_hat[0][0]) / np.sqrt(
    np.correlate(X[0][0], X[0][0]) * np.correlate(X_hat[0][0], X_hat[0][0])
)

###############################################################################
# # Multichannel signals

###############################################################################
# DiCoDiLe works just as well with multi-channel signals. The gait dataset
# contains 16 signals (8 for each foot), in the rest of this tutorial, we’ll
# use three of those.

# Left foot Vertical acceleration, Y rotation and X acceleration
channels = ['LAV', 'LRY', 'LAX']

###############################################################################
# Let’s look at a small portion of multi-channel data

colors = plt.rcParams["axes.prop_cycle"]()
mc_fig, mc_ax = plt.subplots(len(channels), sharex=True)

for ax, chan in zip(mc_ax, channels):
    ax.plot(trial['data'][chan][5000:5800],
            label=chan, color=next(colors)["color"])
mc_fig.legend(loc="upper center")

###############################################################################
# Let’s put the data in shape for alphacsc: (n_trials, n_channels, n_times)

X_mc_subset = trial['data'][channels].to_numpy().T

X_mc_subset = X_mc_subset.reshape(1, *X_mc_subset.shape)

print(X_mc_subset.shape)

###############################################################################
# Initialize the dictionary (note that the call is identical to the
# single-channel version)

D_init_mc = init_dictionary(
    X_mc_subset, n_atoms=8, n_times_atom=200, rank1=False,
    window=True, D_init='chunk', random_state=60
)

print(D_init_mc.shape)

###############################################################################
# And run DiCoDiLe (note that the call is identical to the single-channel
# version here as well)

from alphacsc import BatchCDL

cdl = BatchCDL(
    # Shape of the dictionary
    n_atoms,
    n_times_atom,
    rank1=False,
    uv_constraint='auto',
    # Number of iteration for the alternate minimization and cvg threshold
    n_iter=3,
    # number of workers to be used for dicodile
    n_jobs=4,
    # solver for the z-step
    solver_z='dicodile',
    solver_z_kwargs={'max_iter': 10000},
    window=True,
    D_init=D_init_mc,
    random_state=60)

res = cdl.fit(X_mc_subset)

###############################################################################
# # Signal reconstruction (multichannel)

###############################################################################
# Now, let’s reconstruct the original signal

from alphacsc.utils.convolution import construct_X_multi

z_hat_mc = res._z_hat

D_hat_mc = res._D_hat

X_hat_mc = construct_X_multi(z_hat_mc, D_hat_mc)

###############################################################################
# Let’s visually compare a small part of the original and reconstructed signal
# along with the activations.

z_hat_mc.shape

""
viz_start_idx = 4000
viz_end_idx = 5800
viz_chan = 2

max_abs = np.max(np.abs(z_hat_mc), axis=-1)
max_abs = max_abs.reshape(z_hat_mc.shape[1], 1)
z_hat_normalized = z_hat_mc / max_abs

fig_hat_mc, ax_hat_mc = plt.subplots(2, figsize=(12, 8))

# plot original and constructed
ax_hat_mc[0].plot(X_mc_subset[0][viz_chan][viz_start_idx:viz_end_idx],
                  label='ORIGINAL')
ax_hat_mc[0].plot(X_hat_mc[0][viz_chan][viz_start_idx:viz_end_idx],
                  label='RECONSTRUCTED')

ax_hat_mc[0].set_xlabel('time (x10ms)')
ax_hat_mc[0].legend()

# plot activations
for idx in range(z_hat_normalized.shape[1]):
    ax_hat_mc[1].stem(z_hat_normalized[0][idx][viz_start_idx:viz_end_idx],
                      linefmt=f"C{idx}-",
                      markerfmt=f"C{idx}o")
