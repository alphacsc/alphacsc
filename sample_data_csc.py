import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from numpy import hamming

import mne
from alphacsc.learn_d_z_multi import learn_d_z_multi, _get_D
from alphacsc.utils import construct_X_multi

n_atoms = 5

# get X
data_path = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample')
raw = mne.io.read_raw_fif(op.join(data_path,
                                  'sample_audvis_filt-0-40_raw.fif'),
                          preload=True)
raw.pick_types(meg='mag')
raw.crop(tmax=30.)  # take only 30 s of data
X = raw[:][0]

# define n_chan, n_times, n_trials
n_chan = X.shape[0]
n_times = int(round(raw.info['sfreq'] * 1.))  # 1. s
n_times_atom = int(round(raw.info['sfreq'] * 0.2))  # 200. ms
n_trials = X.shape[-1] // n_times


# make windows
X = X[:, :n_trials * n_times]
X = X.reshape((n_chan, n_trials, n_times)).swapaxes(0, 1)
X *= hamming(n_times)[None, None, :]
X /= np.linalg.norm(X, axis=-1, keepdims=True)

plt.close('all')
fig, axes = plt.subplots(nrows=1, num='atoms', figsize=(10, 8))
fig_topo, axes_topo = plt.subplots(1, n_atoms, figsize=(12, 3))


def callback(X, uv_hat, Z_hat, reg):
    for idx in range(n_atoms):
        mne.viz.plot_topomap(uv_hat[idx, :n_chan], raw.info,
                             axes=axes_topo[idx], show=False)
        axes_topo[idx].set_title('atom %d' % idx)
    if axes.lines == []:
        lines = axes.plot(uv_hat[:, n_chan:].T)
        axes.grid(True)
        axes.set_title('temporal atom')
        axes.legend(lines, ['atom %d' % idx for idx in range(n_atoms)])
    else:
        for line_0, uv in zip(axes.lines, uv_hat):
            line_0.set_ydata(uv[n_chan:])
        axes.relim()  # make sure all the data fits
        axes.autoscale_view(True, True, True)
    plt.draw()
    plt.pause(0.001)


pobj, times, uv_hat, Z_hat = learn_d_z_multi(X, n_atoms, n_times_atom,
                                             random_state=42,
                                             n_jobs=1, reg=0.001,
                                             callback=callback)

D_hat = _get_D(uv_hat, n_chan)
X_hat = construct_X_multi(Z_hat, D_hat)

plt.figure("X")
plt.plot(X[0, 0])
plt.plot(X_hat[0, 0])
plt.show()
