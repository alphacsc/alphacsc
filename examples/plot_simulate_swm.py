"""
=====================
SWM on simulated data
=====================
This example shows how the sliding window method (SWM) [1]
works on simulated data.

[1] Gips, Bart, et al.
"Discovering recurring patterns in electrophysiological recordings."
Journal of neuroscience methods 275 (2017): 66-79.
"""

# Authors: Scott Cole
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt

from alphacsc import check_random_state # noqa
from alphacsc.simulate import simulate_data
from alphacsc.swm import sliding_window_matching


n_times_atom = 64  # L
n_times = 5000  # T
n_atoms = 1  # K
n_trials = 10  # N


random_state_simulate = 1
X, ds_true, Z_true = simulate_data(n_trials, n_times, n_times_atom,
                                   n_atoms, random_state_simulate)

rng = check_random_state(random_state_simulate)
X += 0.01 * rng.randn(*X.shape)

X = X.reshape(X.shape[0] * X.shape[1])

L = 64
G = 1000
random_state = 42

d_hat, window_starts, J = sliding_window_matching(
    X, L, G, max_iterations=10000, T=0.01, random_state=random_state)
plt.plot(d_hat)

plt.figure()
plt.plot(J)
plt.title('Cost function J')
plt.show()
