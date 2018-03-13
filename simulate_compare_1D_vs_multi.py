import matplotlib.pyplot as plt
import numpy as np

from alphacsc.simulate import get_atoms, get_activations
from alphacsc.utils import construct_X_multi_uv
from alphacsc.update_d_multi import prox_uv
from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.learn_d_z import learn_d_z

# Generate synchronous D
n_times_atom, n_times = 64, 512
n_chan = 100
n_atoms = 2
n_trials = 100
n_iter = 100

v0 = get_atoms('triangle', n_times_atom)  # temporal atoms
v1 = get_atoms('square', n_times_atom)

u0 = get_atoms('sin', n_chan)  # spatial maps
u1 = get_atoms('cos', n_chan)

uv = np.array([np.r_[u0, v0], np.r_[u1, v1]])
uv = prox_uv(uv, 'separate', n_chan)

# add atoms
rng = np.random.RandomState(27)
shape_Z = (n_atoms, n_trials, n_times_atom)
Z = get_activations(rng, shape_Z)

X = construct_X_multi_uv(Z, uv, n_chan)
X += 0.01 * rng.randn(*X.shape)

reg = 0.01
random_state = 60

pobj, times, uv_hat, Z_hat = learn_d_z_multi(
    X, n_atoms, n_times_atom, random_state=random_state, callback=None,
    n_iter=n_iter, n_jobs=1, reg=reg, uv_constraint='separate',
    solver_d='alternate',
    solver_d_kwargs={'momentum': False, 'max_iter': 1000})

select_ch = [10]
pobj, times, d_hat, Z_hat = learn_d_z(
    X[:, select_ch, :].squeeze(), n_atoms, n_times_atom,
    reg=reg, n_iter=n_iter,
    solver_d_kwargs=dict(factr=100), random_state=random_state,
    n_jobs=1, verbose=1)

plt.plot(uv_hat[:, n_chan:].T, 'g', label='Multivariate')
plt.plot(d_hat.T, 'r', label='1D')
plt.plot(uv[:, n_chan:].T, 'k--', label='ground truth')
plt.legend()
plt.show()
