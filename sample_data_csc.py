import argparse
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from scipy.signal import tukey

import mne
from alphacsc.learn_d_z_multi import learn_d_z_multi, _get_D
from alphacsc.utils import construct_X_multi


parser = argparse.ArgumentParser('Programme to launch experiment on multi csc')
parser.add_argument('--profile', action='store_true',
                    help='Print profiling of the function')
args = parser.parse_args()

n_atoms = 5

# get X
data_path = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample')
raw = mne.io.read_raw_fif(op.join(data_path,
                                  'sample_audvis_filt-0-40_raw.fif'),
                          preload=True)
raw.pick_types(meg='mag')
raw_data = raw[:][0]
raw.crop(tmax=30.)  # take only 30 s of data
raw.pick_types(meg='mag')
X = raw[:][0]

# define n_chan, n_times, n_trials
n_chan, n_times = X.shape
n_times_atom = int(round(raw.info['sfreq'] * 0.4))  # 400. ms

# make windows
X = X[None, ...]
X *= tukey(n_times, alpha=0.1)[None, None, :]
X /= np.linalg.norm(X, axis=-1, keepdims=True)

plt.close('all')
fig, axes = plt.subplots(nrows=n_atoms, num='atoms', figsize=(10, 8))
fig_Z, axes_Z = plt.subplots(nrows=n_atoms, num='Z', figsize=(10, 8),
                             sharex=True, sharey=True)
fig_topo, axes_topo = plt.subplots(1, n_atoms, figsize=(12, 3))
if n_atoms == 1:
    axes, axes_topo, axes_Z = [axes_topo], [axes], [axes_Z]


def callback(X, uv_hat, Z_hat, reg):
    for idx in range(n_atoms):
        mne.viz.plot_topomap(uv_hat[idx, :n_chan], raw.info,
                             axes=axes_topo[idx], show=False)
        axes_topo[idx].set_title('atom %d' % idx)
    if axes[0].lines == []:
        for k in range(n_atoms):
            axes[k].plot(uv_hat[k, n_chan:].T)
            axes[k].grid(True)
    else:
        for ax, uv in zip(axes, uv_hat):
            ax.lines[0].set_ydata(uv[n_chan:])
            ax.relim()  # make sure all the data fits
            ax.autoscale_view(True, True, True)
    if axes_Z[0].lines == []:
        for k in range(n_atoms):
            axes_Z[k].plot(Z_hat[k, 0])
            axes_Z[k].grid(True)
    else:
        for ax, z in zip(axes_Z, Z_hat[:, 0]):
            ax.lines[0].set_ydata(z)
            ax.relim()  # make sure all the data fits
            ax.autoscale_view(True, True, True)
    fig.canvas.draw()
    fig_topo.canvas.draw()
    fig_Z.canvas.draw()
    plt.pause(.001)


if args.profile:
    import cProfile
    callback = None
    pr = cProfile.Profile()
    pr.enable()
pobj, times, uv_hat, Z_hat = learn_d_z_multi(
    X, n_atoms, n_times_atom, random_state=42, n_iter=100, n_jobs=1, reg=0.1,
    eps=1e-3, solver_z_kwargs={'factr': 1e12},
    solver_d_kwargs={'max_iter': 300}, uv_constraint='separate',
    solver_d='alternate', callback=callback)

if args.profile:
    pr.disable()
    pr.dump_stats('.profile')

plt.figure("Final atom")
plt.plot(uv_hat[0, n_chan:])

D_hat = _get_D(uv_hat, n_chan)
X_hat = construct_X_multi(Z_hat, D_hat)

plt.figure("X")
plt.plot(X.mean(axis=1)[0])
plt.plot(X_hat.mean(axis=1)[0])
plt.show()
