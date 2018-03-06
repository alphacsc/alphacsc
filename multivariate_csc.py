import io
import pstats
import argparse
import cProfile
import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt

from alphacsc.simulate import get_atoms
from alphacsc.utils import construct_X_multi, _get_D
from alphacsc.learn_d_z_multi import learn_d_z_multi


parser = argparse.ArgumentParser('Programme to launch experiment on multi csc')
parser.add_argument('--profile', action='store_true',
                    help='Print profiling of the function')
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

reg = n_chan * 0.001

v0 = get_atoms('triangle', n_times_atom)  # temporal atoms
v1 = get_atoms('square', n_times_atom)

u0 = get_atoms('sin', n_chan)  # spatial maps
u1 = get_atoms('cos', n_chan)

# Build D and scale atoms
atoms = []
for (u, v) in [(u0, v0), (u1, v1)]:
    scale = max(1., norm(np.concatenate([u, v])))
    u /= scale
    v /= scale
    atoms.append(np.outer(u, v))

D = np.array(atoms)
uv = np.array([np.r_[u0, v0], np.r_[u1, v1]])

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

print(Z.shape, D.shape)


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


X = construct_X_multi(Z, D)

pobjs, uv_hats = list(), list()

if args.profile:
    callback = None
    n_states = 1
    n_iter = 10
    pr = cProfile.Profile()
    pr.enable()
for random_state in range(n_states):
    pobj, times, uv_hat, Z_hat = learn_d_z_multi(X, n_atoms, n_times_atom,
                                                 random_state=random_state,
                                                 callback=callback,
                                                 n_iter=n_iter, n_jobs=1,
                                                 reg=reg)
    pobjs.append(pobj[-1])
    uv_hats.append(uv_hat)

if args.profile:
    pr.disable()
    pr.dump_stats('.profile')

    # s = io.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats('alphacsc/*')
    # print(s.getvalue())
    import sys
    sys.exit(0)

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
D_hat = _get_D(uv_hats[best_state], n_chan)
for i, d_hat in enumerate(D_hat):
    plt.subplot(2, 1, i + 1)
    plt.plot(d_hat.T, 'r')
    plt.plot(D[i].T, 'k--')

X_hat = construct_X_multi(Z_hat, D_hat)

plt.figure("X")
plt.plot(X[0, 0])
plt.plot(X_hat[0, 0])
plt.show()
