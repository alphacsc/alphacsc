import argparse
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from scipy.signal import tukey

import mne
from mne.utils import _reject_data_segments
from mne.preprocessing import ICA, create_eog_epochs
from mne import EvokedArray

from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.utils import construct_X_multi, _choose_convolve
from alphacsc.utils import plot_callback, get_D

parser = argparse.ArgumentParser('Programme to launch experiment on multi csc')
parser.add_argument('--profile', action='store_true',
                    help='Print profiling of the function')
args = parser.parse_args()

dataset = 'sample'

n_atoms = 25

# get X
data_path = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample')
raw = mne.io.read_raw_fif(op.join(data_path,
                                  'sample_audvis_filt-0-40_raw.fif'),
                          preload=True)

raw.pick_types(meg='mag', eog=True)
raw.filter(1., 40.)
raw_data = raw[:][0]
raw.crop(tmax=30.)  # take only 30 s of data

# ICA for comparison
picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                           stim=False, exclude='bads')

method = 'fastica'
decim = 3
random_state = 23

eog_inds = [4]

reject = dict(mag=5e-12)
ica = ICA(n_components=n_atoms, method=method, random_state=random_state)
ica.fit(raw, picks=picks_meg, decim=decim, reject=reject)
eog_average = create_eog_epochs(raw, reject=reject, picks=picks_meg).average()

# uncomment next two lines to automatically compute eog_inds
# eog_epochs = create_eog_epochs(raw, reject=reject)  # get single EOG trials
# eog_inds, scores = ica.find_bads_eog(eog_epochs)  # find via correlation

# Now multicsc
raw.pick_types(meg='mag')
X = raw[:][0]
X, _ = _reject_data_segments(X, reject, flat=None, decim=None,
                             info=raw.info, tstep=0.3)

# define n_chan, n_times, n_trials
n_channels, n_times = X.shape
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
    X, n_atoms, n_times_atom, random_state=42, n_iter=60, n_jobs=1, reg=1e-2,
    eps=1e-3, solver_z_kwargs={'factr': 1e12},
    solver_d_kwargs={'max_iter': 300}, uv_constraint='separate',
    solver_d='alternate_adaptive', callback=callback)

if args.profile:
    pr.disable()
    pr.dump_stats('.profile')

X_hat = construct_X_multi(Z_hat, uv_hat, n_channels=n_channels)

plt.figure("X")
plt.plot(X.mean(axis=1)[0])
plt.plot(X_hat.mean(axis=1)[0])
plt.show()

# Look at v * Z for one trial
# (we have only one trial, so full time series)
X_hat_k = np.zeros((n_atoms, n_times))
for k in range(n_atoms):
    X_hat_k[k] = _choose_convolve(Z_hat[k, 0, :][None, :],
                                  uv_hat[k, n_channels:][None, :])
ch_names = ['atom %d' % ii for ii in range(n_atoms)]
info = mne.create_info(ch_names, sfreq=raw.info['sfreq'])
raw_atoms = mne.io.RawArray(X_hat_k, info, first_samp=raw.first_samp)
raw_atoms.plot(scalings=dict(misc='auto'))

# Plot interesting atoms
atoms_idx = [14, 18, 23]
times = np.arange(n_times_atom) / raw.info['sfreq']
fig, axes = plt.subplots(1, len(atoms_idx), figsize=(12, 4))
for idx, atom_idx in enumerate(atoms_idx):
    mne.viz.plot_topomap(uv_hat[atom_idx, :n_channels], raw.info,
                         axes=axes[idx])
    axes[idx].set_title('atom%d' % atom_idx)
fig.canvas.set_window_title('Spatial atom')

fig, axes = plt.subplots(len(atoms_idx), 1, sharex=True, sharey=True)
fig.canvas.set_window_title('Temporal atom')
for idx, atom_idx in enumerate(atoms_idx):
    axes[idx].plot(times, uv_hat[atoms_idx[idx], n_channels:].T)
    axes[idx].set_title('Atom%d' % atom_idx)

# XXX: what is this 20 Hz atom? It doesn't have topomap of motor ...
atom_idx = 14
plt.figure('Power spectral density')
sfreq = raw.info['sfreq']
psd = np.abs(np.fft.rfft(uv_hat[atom_idx, :n_channels])) ** 2
freqs = np.linspace(0, sfreq / 2.0, psd.shape[0])
plt.plot(freqs, psd.T)
plt.gca().set(xscale='log')

D = get_D(uv_hat, n_channels)[atom_idx]
evoked = EvokedArray(D, raw.info)
evoked.save('examples_multicsc/atom_multi_sample-ave.fif')

ica.plot_properties(raw, picks=eog_inds, psd_args={'fmax': 35.},
                    image_args={'sigma': 1.})
ica.plot_sources(raw, exclude=eog_inds)  # look at source time course
