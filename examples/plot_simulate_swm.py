"""
=====================
SWM on simulated data
=====================
This example shows how the sliding window method (SWM) [1]
works on simulated data.

"""

import matplotlib.pyplot as plt

from alphacsc.simulate import simulate_data
from alphacsc.swm import sliding_window_matching


n_times_atom = 64  # L
n_times = 5000  # T
n_atoms = 2  # K
n_trials = 10  # N


random_state_simulate = 1
X, ds_true, Z_true = simulate_data(n_trials, n_times, n_times_atom,
                                   n_atoms, random_state_simulate)
X = X.reshape(X.shape[0] * X.shape[1])

Fs = 1
L = 64
G = 1000

d_hat, window_starts, J = sliding_window_matching(X, Fs, L, G,
                                                  max_iterations=1000)
plt.plot(d_hat)
plt.show()
