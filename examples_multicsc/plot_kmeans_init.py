import os
import time
from functools import partial

import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals.joblib import Memory, Parallel, delayed
from scipy.signal import tukey

from alphacsc.init_dictionaryict import init_dictionary

figure_path = 'figures'
mem = Memory(cachedir='.', verbose=0)
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

sfreq = 300.
n_times_atom = int(round(sfreq * 0.3))  # 300. ms
n_atoms = 5
n_jobs = 1


@mem.cache()
def load_data(sfreq=sfreq):
    data_path = os.path.join(mne.datasets.somato.data_path(), 'MEG', 'somato')
    raw = mne.io.read_raw_fif(
        os.path.join(data_path, 'sef_raw_sss.fif'), preload=True)
    raw.filter(2., 90., n_jobs=n_jobs)
    raw.notch_filter(np.arange(50, 101, 50), n_jobs=n_jobs)

    events = mne.find_events(raw, stim_channel='STI 014')

    event_id, tmin, tmax = 1, -1., 3.
    baseline = (None, 0)
    picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                           stim=False)

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        picks=picks, baseline=baseline, reject=dict(
                            grad=4000e-13, eog=350e-6), preload=True)
    epochs.pick_types(meg='grad', eog=False)
    epochs.resample(sfreq, npad='auto')

    # define n_channels, n_trials, n_times
    X = epochs.get_data()
    n_trials, n_channels, n_times = X.shape
    X *= tukey(n_times, alpha=0.1)[None, None, :]
    X /= np.std(X)
    return X


all_func = [
    partial(init_dictionary, D_init='random'),
    partial(init_dictionary, D_init='chunk'),
    partial(init_dictionary, D_init='kmeans', D_init_params=dict(
        max_iter=0, distances='euclidean')),
    partial(init_dictionary, D_init='kmeans', D_init_params=dict(
        max_iter=100, distances='euclidean')),
    partial(init_dictionary, D_init='kmeans', D_init_params=dict(
        max_iter=0, distances='roll_inv')),
    partial(init_dictionary, D_init='kmeans', D_init_params=dict(
        max_iter=100, distances='roll_inv')),
    partial(init_dictionary, D_init='kmeans', D_init_params=dict(
        max_iter=0, distances='trans_inv')),
    partial(init_dictionary, D_init='kmeans', D_init_params=dict(
        max_iter=100, distances='trans_inv')),
]

labels = [
    'random',
    'chunk',
    'kmeans_eucl_0',
    'kmeans_eucl_100',
    'kmeans_roll_0',
    'kmeans_roll_100',
    'kmeans_trans_0',
    'kmeans_trans_100',
]

X = load_data()
n_trials, n_channels, n_times = X.shape


def one_run(method, label):
    start = time.time()
    res = method(X, n_atoms, n_times_atom, random_state=0)
    duration = time.time() - start
    print('%s : %.3f sec' % (label, duration))
    return res


save_name = os.path.join(figure_path, 'somato')

results = Parallel(n_jobs=n_jobs)(delayed(one_run)(func, label)
                                  for func, label in zip(all_func, labels))

# ------------ plot the atoms init
fig, axes = plt.subplots(
    len(results), n_atoms, sharex=True, sharey=True,
    figsize=(2 + n_atoms * 3, len(results) * 3))
axes = np.atleast_1d(axes).reshape(len(results), n_atoms)
for i_func, (label, D_init) in enumerate(zip(labels, results)):
    v_init = D_init[:, n_channels:]
    for i_atom in range(n_atoms):
        ax = axes[i_func, i_atom]
        ax.plot(
            np.arange(v_init.shape[1]) / sfreq, v_init[i_atom],
            color=colors[i_atom % len(colors)])
        ax.set_xlabel('Time (s)')
        ax.set_title(label)
        ax.grid('on')
plt.tight_layout()
fig.savefig(save_name + '_atoms_init.png')

plt.close('all')
# plt.show()
