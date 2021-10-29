"""
====================
Gait (steps) example
====================
In this example, we use DiCoDiLe on an open dataset of gait (steps) IMU time-series to discover patterns in the data. We will then use those to attempt to detect steps and compare our findings with the ground truth.
"""

###############################################################################
import matplotlib.pyplot as plt
import numpy as np
from dicodile.data.gait import get_gait_data

from alphacsc.utils import construct_X_multi
from alphacsc import learn_d_z_multi


###############################################################################
# # Retrieve trial data

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
# Now, let’s use DiCoDiLe to learn patterns from the data and reconstruct the
# signal from a sparse representation.
#
# First, we initialize a dictionary from parts of the signal:

X = trial['data']['RAV'].to_numpy()

# reshape X to (n_trials, n_channels, n_times)
X = X.reshape(1, 1, *X.shape)

X.shape

###############################################################################
# Note the use of reshape to shape the signal as per alphacsc requirements: the
# shape of the signal should be (n_trials, n_channels, n_times). Here, we have
# a single-channel time series so it is (1, 1, n_times).

# set dictionary size
n_atoms = 8

# set individual atom (patch) size
n_times_atom = 200

""

solver_z_kwargs = {}

solver_z_kwargs["max_iter"] = 100000

pobj, times, D_hat, z_hat, reg = learn_d_z_multi(X,
                                                 n_atoms,
                                                 n_times_atom,
                                                 n_jobs=4,
                                                 solver_z='dicodile',
                                                 rank1=False,
                                                 loss_params=None,
                                                 window=True,
                                                 n_iter=3,
                                                 random_state=60,
                                                 solver_z_kwargs=solver_z_kwargs)
