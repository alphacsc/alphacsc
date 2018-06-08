from scipy.signal import tukey

import numpy as np
import matplotlib.pyplot as plt

from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.utils import plot_callback

n_atoms = 5

tmin, tmax = -1., 4.
event_id = dict(hands=2, feet=3)
subject = 1
runs = [6, 10, 14]  # motor imagery: hands vs feet

raw_fnames = eegbci.load_data(subject, runs)
raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in
             raw_fnames]
raw = concatenate_raws(raw_files)

raw.rename_channels(lambda x: x.strip('.'))

layout = read_layout('EEG1005')

raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

events = find_events(raw, shortest_event=0, stim_channel='STI 014')

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)
# epochs_train = epochs.copy().crop(tmin=1., tmax=2.)

n_times_atom = int(round(0.3 * raw.info['sfreq']))
X = epochs.get_data()
n_trials, n_channels, n_times = X.shape
X *= tukey(n_times, alpha=0.1)[None, None, :]
X /= np.std(X)


plt.close('all')
callback = plot_callback(X, epochs.info, n_atoms, layout=layout)


pobj, times, uv_hat, Z_hat = learn_d_z_multi(
    X, n_atoms, n_times_atom, random_state=42, n_iter=60, n_jobs=1, reg=5e-2,
    eps=1e-10, solver_z_kwargs={'factr': 1e12},
    solver_d_kwargs={'max_iter': 300}, uv_constraint='separate',
    solver_d='alternate_adaptive', callback=callback)
