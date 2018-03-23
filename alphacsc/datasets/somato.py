
import os
import numpy as np
from scipy.signal import tukey

import mne
from sklearn.externals.joblib import Memory


mem = Memory(cachedir='.', verbose=0)


@mem.cache()
def load_data(sfreq, n_jobs=1):
    data_path = os.path.join(mne.datasets.somato.data_path(), 'MEG', 'somato')
    raw = mne.io.read_raw_fif(
        os.path.join(data_path, 'sef_raw_sss.fif'), preload=True)
    raw.notch_filter(np.arange(50, 101, 50), n_jobs=n_jobs)
    raw.filter(2., None, n_jobs=n_jobs)

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

    # define n_chan, n_trials, n_times
    X = epochs.get_data()
    n_trials, n_chan, n_times = X.shape
    X *= tukey(n_times, alpha=0.1)[None, None, :]
    X /= np.std(X)
    return X
