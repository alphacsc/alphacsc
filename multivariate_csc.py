import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm

from alphacsc.simulate import get_atoms
from alphacsc.utils import construct_X_multi, _get_D
from alphacsc.learn_d_z_multi import learn_d_z_multi


# Generate synchronous D
n_times_atom, n_times = 20, 500
n_chan = 5
n_atoms = 2
n_trials = 10

v0 = get_atoms('triangle', n_times_atom)  # temporal atoms
v1 = get_atoms('square', n_times_atom)

u0 = np.random.uniform(size=n_chan)  # spatial maps
u1 = np.random.uniform(size=n_chan)

uv = np.array([np.r_[u0, v0], np.r_[u1, v1]])
uv /= norm(uv)

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

pobjs, uv_hats = list(), list()
for random_state in range(7):
    pobj, times, uv_hat, Z_hat = learn_d_z_multi(X, n_atoms, n_times_atom,
                                                 random_state=random_state)
    pobjs.append(pobj[-1])
    uv_hats.append(uv_hat)

best_state = np.argmin(pobjs)
plt.plot(pobj)

plt.figure("u (Spatial maps)")
plt.plot(uv_hats[best_state][:, :n_chan].T, 'b')
plt.plot(uv[:, :n_chan].T, 'b--')

plt.figure("v (Temporal atoms)")
plt.plot(uv_hats[best_state][:, n_chan:].T, 'r')
plt.plot(uv[:, n_chan:].T, 'r--')

plt.figure("D")
D_hat = _get_D(uv_hats[best_state], n_chan)
for i, d_hat in enumerate(D_hat):
    plt.subplot(2, 1, i + 1)
    plt.plot(d_hat.T, '--')
    plt.plot(D[i].T)

X_hat = construct_X_multi(Z_hat, D_hat)

plt.figure("X")
plt.plot(X[0, 0])
plt.plot(X_hat[0, 0])
plt.show()
