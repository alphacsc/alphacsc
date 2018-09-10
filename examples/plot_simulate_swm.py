"""
=====================
SWM on simulated data
=====================

This example shows how the sliding window method (SWM) [1]
works on simulated data. The code is adapted from the
`neurodsp package <https://github.com/voytekresearch/neurodsp/>`_
from Voytek lab. Note that, at present, it does not
implement parallel tempering.

[1] Gips, Bart, et al.
    Discovering recurring patterns in electrophysiological recordings.
    Journal of neuroscience methods 275 (2017): 66-79.
"""

# Authors: Scott Cole
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

###############################################################################
# Let us define the model parameters

n_times_atom = 64  # L
n_times = 5000  # T
n_trials = 10  # N

###############################################################################
# The algorithm does not naturally lend itself to multiple atoms. Therefore,
# we simulate only one atom.
n_atoms = 1  # K

###############################################################################
# A minimum spacing between the windows averaged must be found.
min_spacing = 200  # G

###############################################################################
# Now, we can simulate
from alphacsc import check_random_state # noqa
from alphacsc.simulate import simulate_data # noqa

random_state_simulate = 1
X, ds_true, z_true = simulate_data(n_trials, n_times, n_times_atom,
                                   n_atoms, random_state_simulate,
                                   constant_amplitude=True)

rng = check_random_state(random_state_simulate)
X += 0.01 * rng.randn(*X.shape)

###############################################################################
# We expect 10 occurences of the atom in total.
# So, let us define 10 random locations for the algorithm to start with.
# If this number is not known, we will end up estimating more/less windows.
import numpy as np # noqa
window_starts = rng.choice(np.arange(n_trials * n_times), size=n_trials)

###############################################################################
# Now, we apply the SWM algorithm now.
from alphacsc.other.swm import sliding_window_matching # noqa

random_state = 42
X = X.reshape(X.shape[0] * X.shape[1])  # expects 1D time series
d_hat, window_starts, J = sliding_window_matching(
    X, L=n_times_atom, G=min_spacing, window_starts_custom=window_starts,
    max_iterations=10000, T=0.01, random_state=random_state)

###############################################################################
# Let us look at the data at the time windows when the atoms are found.
import matplotlib.pyplot as plt # noqa
fig, axes = plt.subplots(2, n_trials // 2, sharex=True, sharey=True,
                         figsize=(15, 3))
axes = axes.ravel()
for ax, w_start in zip(axes, window_starts):
    ax.plot(X[w_start:w_start + n_times_atom])

###############################################################################
# It is not perfect, but it does find time windows where the atom
# is present. Now let us plot the atoms.

plt.figure()
plt.plot(d_hat / np.linalg.norm(d_hat))
plt.plot(ds_true.T, '--')

###############################################################################
# and the cost function over iterations

plt.figure()
plt.plot(J)
plt.ylabel('Cost function J')
plt.xlabel('Iteration #')
plt.show()
