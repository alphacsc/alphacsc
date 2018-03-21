import argparse
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from scipy.signal import tukey

import mne
from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.utils import construct_X_multi_uv, plot_callback


parser = argparse.ArgumentParser('Programme to launch experiment on multi csc')
parser.add_argument('--profile', action='store_true',
                    help='Print profiling of the function')
args = parser.parse_args()

dataset = 'sample'

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
callback = plot_callback(X, raw.info, n_atoms)

if args.profile:
    import cProfile
    callback = None
    pr = cProfile.Profile()
    pr.enable()
pobj, times, uv_hat, Z_hat = learn_d_z_multi(
    X, n_atoms, n_times_atom, random_state=42, n_iter=60, n_jobs=1, reg=2e-2,
    eps=1e-3, solver_z_kwargs={'factr': 1e12},
    solver_d_kwargs={'max_iter': 300}, uv_constraint='separate',
    solver_d='alternate_adaptive', callback=callback)

if args.profile:
    pr.disable()
    pr.dump_stats('.profile')

plt.figure("Final atom")
plt.plot(uv_hat[0, n_chan:])

X_hat = construct_X_multi_uv(Z_hat, uv_hat, n_chan)

plt.figure("X")
plt.plot(X.mean(axis=1)[0])
plt.plot(X_hat.mean(axis=1)[0])
plt.show()
