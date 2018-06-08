import numpy as np
import matplotlib.pyplot as plt

from multicsc import check_random_state
from multicsc.simulate import simulate_data
from multicsc.learn_d_z_multi import learn_d_z_multi
from multicsc.learn_d_z import learn_d_z

###############################################################################
n_times_atom = 64  # L
n_times = 512  # T
n_atoms = 2  # K
n_trials = 100  # N
n_iter = 50

reg = 0.1

random_state_simulate = 1
X, ds_true, Z_true = simulate_data(n_trials, n_times, n_times_atom, n_atoms,
                                   random_state_simulate)

rng = check_random_state(random_state_simulate)
X += 0.01 * rng.randn(*X.shape)

n_times_atom += 32
###############################################################################
# Multichannel CSC
n_channels = 1
fig, axes = plt.subplots(nrows=2, num='atoms', figsize=(10, 8))


def callback(X, uv_hat, Z_hat, reg):
    plt.figure('atoms')
    if axes[0].lines == []:
        axes[0].plot(Z_hat.sum(axis=1).T)
        axes[1].plot(uv_hat[:, n_channels:].T)
        axes[0].grid(True)
        axes[1].grid(True)
        axes[0].set_title('activations')
        axes[1].set_title('temporal atom')
    else:
        for line_0, line_1, uv, z in zip(axes[0].lines, axes[1].lines, uv_hat,
                                         Z_hat):
            line_0.set_ydata(z.sum(axis=0))
            line_1.set_ydata(uv[n_channels:])
    for ax in axes:
        ax.relim()  # make sure all the data fits
        ax.autoscale_view(True, True, True)
    plt.draw()
    plt.pause(0.001)


n_iter = 20
random_state = 60

reg_ = reg

rng = check_random_state(random_state)
ds_init = rng.randn(n_atoms, n_times_atom)
D_init = np.hstack([np.ones((n_atoms, 1)), ds_init])

if True:
    pobj, times, uv_hat, Z_hat = learn_d_z_multi(
        X[:, None, :], n_atoms, n_times_atom, reg=reg_, n_iter=n_iter,
        solver_z_kwargs={'factr': 1e10}, random_state=random_state, n_jobs=1,
        verbose=1, callback=callback, solver_d='joint',
        uv_constraint='separate')
    print('Multichannel CSC')
    print(pobj)

    plt.figure()
    plt.plot(uv_hat[:, 1:].T)
    plt.plot(ds_true.T, 'k--')
    plt.show()

###############################################################################
# Vanilla CSC

fig, axes = plt.subplots(nrows=2, num='atoms', figsize=(10, 8))


def callback(X, uv_hat, Z_hat, reg):
    if axes[0].lines == []:
        axes[0].plot(Z_hat.sum(axis=1).T)
        axes[1].plot(uv_hat[:, :].T)
        axes[0].grid(True)
        axes[1].grid(True)
        axes[0].set_title('activations')
        axes[1].set_title('temporal atom')
    else:
        for line_0, line_1, uv, z in zip(axes[0].lines, axes[1].lines, uv_hat,
                                         Z_hat):
            line_0.set_ydata(z.sum(axis=0))
            line_1.set_ydata(uv[:])
    for ax in axes:
        ax.relim()  # make sure all the data fits
        ax.autoscale_view(True, True, True)
    plt.draw()
    plt.pause(0.001)


if True:
    pobj, times, d_hat, Z_hat = learn_d_z(
        X, n_atoms, n_times_atom, reg=reg, n_iter=n_iter, solver_d_kwargs=dict(
            factr=100), random_state=random_state, n_jobs=1, verbose=1,
        ds_init=ds_init, callback=callback)
    print('Vanilla CSC')
    print(pobj)

    plt.figure()
    plt.plot(d_hat.T)
    plt.plot(ds_true.T, 'k--')
    plt.show()
