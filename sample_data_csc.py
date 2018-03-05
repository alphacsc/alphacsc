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
raw = mne.io.read_raw_fif(op.join(data_path, 'sample_audvis_raw.fif'),
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
