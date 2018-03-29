import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from scipy.signal import tukey

import mne
from mne.utils import _reject_data_segments
from mne.preprocessing import compute_proj_ecg, compute_proj_eog

from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.utils import (construct_X_multi_uv, _choose_convolve,
                            plot_callback)

# Trying to do something like:
# https://martinos.org/mne/stable/auto_examples/decoding/plot_ems_filtering.html#sphx-glr-auto-examples-decoding-plot-ems-filtering-py

dataset = 'sample'

n_atoms = 10
# reject = dict(mag=5e-12)

decim = 3
random_state = 23

event_id = {'Auditory/Left': 1, 'Auditory/Right': 2,
            'Visual/Left': 3, 'Visual/Right': 4,
            'smiley': 5, 'button': 32}

plt.close('all')

# get X
data_path = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample')
raw = mne.io.read_raw_fif(op.join(data_path,
                                  'sample_audvis_filt-0-40_raw.fif'),
                          preload=True)

# remove physiological artifacts
projs, events = compute_proj_ecg(raw, n_grad=1, n_mag=1, n_eeg=0, average=True)
ecg_projs = projs[-2:]
projs, events = compute_proj_eog(raw, n_grad=1, n_mag=1, n_eeg=1, average=True)
eog_projs = projs[-3:]
# uncomment next lines to visualize projs
# mne.viz.plot_projs_topomap(ecg_projs)
# mne.viz.plot_projs_topomap(eog_projs, info=raw.info)
raw.info['projs'] += eog_projs + ecg_projs
events = mne.find_events(raw)

raw.apply_proj()

raw.pick_types(meg='grad', eog=True)
raw.filter(1., None)
raw_data = raw[:][0]
# raw.crop(tmax=30.)  # take only 30 s of data

picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                           stim=False, exclude='bads')

# Now multicsc
raw.pick_types(meg='grad')
X = raw[:][0]
# X, _ = _reject_data_segments(X, reject, flat=None, decim=None,
#                              info=raw.info, tstep=0.3)

# define n_chan, n_times, n_trials
n_chan, n_times = X.shape
n_times_atom = int(round(raw.info['sfreq'] * 0.1))  # 100. ms

epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.3, tmax=0.7,
                    reject=dict(grad=4000e-13))
epochs = epochs[['Auditory/Left', 'Visual/Left']]

# make windows
# X = X[None, ...]
X = epochs.get_data()
n_trials, n_chan, n_times = X.shape
X *= tukey(n_times, alpha=0.1)[None, None, :]
X /= np.std(X)

plt.close('all')
callback = plot_callback(X, raw.info, n_atoms)

pobj, times, uv_hat, Z_hat = learn_d_z_multi(
    X, n_atoms, n_times_atom, random_state=42, n_iter=60, n_jobs=1, reg=5e-2,
    eps=1e-10, solver_z_kwargs={'factr': 1e12},
    solver_d_kwargs={'max_iter': 300}, uv_constraint='separate',
    solver_d='alternate_adaptive', callback=callback)

X_hat = construct_X_multi_uv(Z_hat, uv_hat, n_chan)

plt.figure("X")
plt.plot(X.mean(axis=1)[0])
plt.plot(X_hat.mean(axis=1)[0])
plt.show()

# Look at v * Z for one trial
# (we have only one trial, so full time series)
X_hat_k = np.zeros((n_atoms, n_times))
for k in range(n_atoms):
    X_hat_k[k] = _choose_convolve(Z_hat[k, 0, :][None, :],
                                  uv_hat[k, n_chan:][None, :])
ch_names = ['atom %d' % ii for ii in range(n_atoms)]
info = mne.create_info(ch_names, sfreq=raw.info['sfreq'])

evoked_aud = epochs['Auditory/Left'].average()
evoked_aud.plot_topomap(times=0.093)  # peak at 93 ms

evoked_vis = epochs['Visual/Left'].average()
evoked_vis.plot_topomap(times=[0.093, 0.186])  # peaks at 93 ms and 186 ms

# raw_atoms = mne.io.RawArray(X_hat_k, info, first_samp=raw.first_samp)

# plot_idx = 8
# plt.figure('spatial map')
# mne.viz.plot_topomap(uv_hat[plot_idx, :n_chan], raw.info)
# plt.figure('temporal atom')
# plt.plot(uv_hat[plot_idx, n_chan:])
