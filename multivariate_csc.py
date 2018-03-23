import argparse
import numpy as np
import matplotlib.pyplot as plt

from alphacsc.simulate import get_atoms, get_activations
from alphacsc.utils import construct_X_multi_uv, get_D
from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.update_d_multi import prox_uv


parser = argparse.ArgumentParser('Programme to launch experiment on multi csc')
parser.add_argument('--momentum', action='store_true',
                    help='Use the momentum for d updates')
parser.add_argument('--n_iter', type=int, default=400,
                    help='Print profiling of the function')

args = parser.parse_args()


# Generate synchronous D
n_times_atom, n_times = 100, 601
n_chan = 100
n_atoms = 2
n_states = 7
n_trials = 30
n_iter = 400

lmbd_max = 'fixed'
if lmbd_max != 'fixed':
    reg = .8
else:
    reg = n_chan * 1e-4

v0 = get_atoms('triangle', n_times_atom)  # temporal atoms
v1 = get_atoms('square', n_times_atom)

u0 = get_atoms('sin', n_chan)  # spatial maps
u1 = get_atoms('cos', n_chan)

# Build D and scale atoms
uv = np.array([np.r_[u0, v0], np.r_[u1, v1]])
uv = prox_uv(uv, 'separate', n_chan)
D = get_D(uv, n_chan)

# add atoms
rng = np.random.RandomState(27)
shape_Z = (n_atoms, n_trials, n_times_atom)
Z = get_activations(rng, shape_Z)


fig, axes = plt.subplots(nrows=2, num='atoms', figsize=(10, 8))


def callback(X, uv_hat, Z_hat, reg):
    if axes[0].lines == []:
        axes[0].plot(uv_hat[:, :n_chan].T)
        axes[1].plot(uv_hat[:, n_chan:].T)
        axes[0].grid(True)
        axes[1].grid(True)
        axes[0].set_title('spatial atom')
        axes[1].set_title('temporal atom')
    else:
        for line_0, line_1, uv in zip(axes[0].lines, axes[1].lines, uv_hat):
            line_0.set_ydata(uv[:n_chan])
            line_1.set_ydata(uv[n_chan:])
    for ax in axes:
        ax.relim()  # make sure all the data fits
        ax.autoscale_view(True, True, True)
    plt.draw()
    plt.pause(0.001)


X = construct_X_multi_uv(Z, uv, n_chan)

pobjs, uv_hats = list(), list()

for random_state in range(n_states):
    pobj, times, uv_hat, Z_hat = learn_d_z_multi(
        X, n_atoms, n_times_atom, random_state=random_state, callback=callback,
        n_iter=n_iter, n_jobs=1, reg=reg, uv_constraint='separate',
        solver_d='alternate_adaptive', solver_z="l_bfgs",
        solver_d_kwargs={'max_iter': 50},
        solver_z_kwargs={'factr': 10e11},
        # uv_init='ssa',
        # lmbd_max=True,
        loss='dtw', loss_params=dict(gamma=0.005, sakoe_chiba_band=10),)
    pobjs.append(pobj[-1])
    uv_hats.append(uv_hat)

best_state = np.argmin(pobjs)

plt.close('all')
plt.plot(pobj)

plt.figure("u (Spatial maps)")
plt.plot(uv_hats[best_state][:, :n_chan].T, 'r')
plt.plot(uv[:, :n_chan].T, 'k--')

plt.figure("v (Temporal atoms)")
plt.plot(uv_hats[best_state][:, n_chan:].T, 'r')
plt.plot(uv[:, n_chan:].T, 'k--')

plt.figure("D")
D_hat = get_D(uv_hats[best_state], n_chan)
for i, d_hat in enumerate(D_hat):
    plt.subplot(2, 1, i + 1)
    plt.plot(d_hat.T, 'r')
    plt.plot(D[i].T, 'k--')

X_hat = construct_X_multi_uv(Z_hat, uv_hats[best_state], n_chan)

plt.figure("X")
plt.plot(X[0, 0])
plt.plot(X_hat[0, 0])
plt.show()
