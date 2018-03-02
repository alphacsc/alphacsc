import numpy as np
import matplotlib.pyplot as plt

from alphacsc.simulate import get_atoms
from alphacsc.utils import _sparse_multi_convolve


# Generate synchronous D
n_times_atom, n_times = 20, 500
n_chan = 5
n_trials = 10

d0 = get_atoms('triangle', n_times_atom)
d1 = get_atoms('square', n_times_atom)

u0 = np.random.uniform(size=n_chan)
u1 = np.random.uniform(size=n_chan)

D = np.array([np.outer(u0, d0), np.outer(u1, d1)])


starts = list()

# add atoms
rng = np.random.RandomState(27)
for _ in range(2):
    starts.append(rng.randint(low=0, high=n_times - n_times_atom + 1,
                  size=(n_trials,)))
# add activations
Z = np.zeros((n_trials, 2, n_times))
for i in range(n_trials):
    for k_idx, (d, start) in enumerate(zip(D, starts)):
        Z[i, k_idx, starts[k_idx][i]] = rng.uniform()


X = [_sparse_multi_convolve(z, D) for z in Z]

plt.plot(X[0].T)
plt.show()
