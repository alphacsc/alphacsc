import os.path as op
import matplotlib.pyplot as plt
from numpy import hamming

import mne
from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.utils import construct_X_multi

n_atoms = 5

# get X
data_path = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample')
raw = mne.io.read_raw_fif(op.join(data_path, 'sample_audvis_raw.fif'),
                          preload=True)
raw.pick_types(meg='mag')
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

pobj, times, uv_hat, Z_hat = learn_d_z_multi(X, n_atoms, n_times_atom,
                                             random_state=42,
                                             n_jobs=1, reg=0.001)

plt.figure("X")
plt.plot(X[0, 0])
plt.plot(X_hat[0, 0])
plt.show()
