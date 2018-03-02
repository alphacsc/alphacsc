import numpy as np
import matplotlib.pyplot as plt

from alphacsc.simulate import get_atoms
from alphacsc.utils import construct_X_multi
from alphacsc.learn_d_z_multi import learn_d_z_multi


# Generate synchronous D
n_times_atom, n_times = 20, 500
n_chan = 5
n_atoms = 2
n_trials = 10

v0 = get_atoms('triangle', n_times_atom)
v1 = get_atoms('square', n_times_atom)

u0 = np.random.uniform(size=n_chan)
u1 = np.random.uniform(size=n_chan)

D = np.array([np.outer(u0, v0), np.outer(u1, v1)])


starts = list()

# add atoms
rng = np.random.RandomState(27)
for _ in range(n_atoms):
    starts.append(rng.randint(low=0, high=n_times - n_times_atom + 1,
                  size=(n_trials,)))
# add activations
Z = np.zeros((n_atoms, n_trials, n_times))
for i in range(n_trials):
    for k_idx, (d, start) in enumerate(zip(D, starts)):
        Z[k_idx, i, starts[k_idx][i]] = rng.uniform()


X = construct_X_multi(Z, D)

pobj, times, u_hat, v_hat, Z_hat = learn_d_z_multi(X, n_atoms, n_times_atom)

plt.plot(times, pobj)
plt.show()


plt.plot(X[0].T)
plt.show()
