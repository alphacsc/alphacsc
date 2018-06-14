import numpy as np
import matplotlib.pyplot as plt

from multicsc.simulate import get_atoms, get_activations
from multicsc.utils import construct_X_multi, get_D
from multicsc.update_d_multi import prox_uv
from multicsc import learn_d_z

from sklearn.decomposition import FastICA

# Generate synchronous D
n_times_atom, n_times = 100, 601
n_channels = 100
n_atoms = 2
n_states = 7
n_trials = 30
n_iter = 100

lmbd_max = 'fixed'
if lmbd_max != 'fixed':
    reg = .8
else:
    reg = n_channels * 1e-4

v0 = get_atoms('triangle', n_times_atom)  # temporal atoms
v1 = get_atoms('square', n_times_atom)

u0 = get_atoms('sin', n_channels)  # spatial maps
u1 = get_atoms('cos', n_channels)
# u0 = np.random.randn(*u0.shape)
# u1 = np.random.randn(*u1.shape)

# Build D and scale atoms
uv = np.array([np.r_[u0, v0], np.r_[u1, v1]])
uv = prox_uv(uv, 'separate', n_channels)
D = get_D(uv, n_channels)

# add atoms
rng = np.random.RandomState(27)
shape_z = (n_atoms, n_trials, n_times - n_times_atom + 1)
z = get_activations(rng, shape_z)

X = construct_X_multi(z, uv, n_channels=n_channels)
X = X.swapaxes(0, 1).reshape((n_channels, -1))

X += np.random.randn(*X.shape) * 0.1 * X.std()

random_state = 3

ica = FastICA(n_components=n_atoms, random_state=random_state,
              tol=1e-12)
XT = ica.fit_transform(X.T)

# plt.plot(XT)
# plt.show()

# plt.plot(ica.components_.T)
# plt.show()
# plt.plot(ica.mixing_)
# plt.show()

pobj, times, v_hat, z_hat = learn_d_z(
    XT[:, 0][None, :], 1, n_times_atom, reg=reg, n_iter=n_iter,
    solver_d_kwargs=dict(factr=100), random_state=random_state,
    n_jobs=1, verbose=1)

plt.plot(v_hat.T, 'r')
plt.plot(np.c_[v0, v1], 'k--')
plt.show()
