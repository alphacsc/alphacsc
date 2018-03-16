import os

import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals.joblib import Memory, Parallel, delayed
from scipy.signal import tukey, spectrogram

figure_path = 'figures'
mem = Memory(cachedir='.', verbose=0)
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

sfreq = 300.
n_times_atom = int(round(sfreq * 0.3))  # 300. ms
reg_list = np.arange(5, 16)

n_atoms = 30
n_iter = 500
n_states = 1
n_jobs = 10

verbose = 1


@mem.cache()
def load_data(sfreq=sfreq):
    data_path = os.path.join(mne.datasets.somato.data_path(), 'MEG', 'somato')
    raw = mne.io.read_raw_fif(
        os.path.join(data_path, 'sef_raw_sss.fif'), preload=True)
    raw.filter(15., 90., n_jobs=n_jobs)
    raw.notch_filter(np.arange(50, 101, 50), n_jobs=n_jobs)

    events = mne.find_events(raw, stim_channel='STI 014')

    event_id, tmin, tmax = 1, -1., 5.
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


def spectro_mean(X_n):
    return np.mean([
        spectrogram(X_np, sfreq, nperseg=32, noverlap=24, nfft=256)[2]
        for X_np in X_n
    ], axis=0)


X = load_data()

# plot the PSD of each trial
if True:
    # pip install git+https://github.com/pactools/pactools.git#egg=pactools
    from pactools.utils.spectrum import Spectrum
    sp = Spectrum(fs=sfreq)
    for X_n in X:
        sp.periodogram(X_n, True, True)
    sp.plot()
    plt.show()

# compute the spectrogram, with a mean over channels
f, t, _ = spectrogram(X[0, 0], sfreq, nperseg=32, noverlap=24, nfft=256)
S = Parallel(n_jobs=5)(delayed(spectro_mean)(X_n)
                       for X_n in X)

# take also the mean over trials
Sxx = np.mean(S, axis=0)

# remove the time edges
Sxx = Sxx[:, 10:-10]
t = t[10:-10]

# plot it
plt.pcolormesh(t, f, Sxx)
plt.show()

# plot the spectrogram variations (normalized frequency by frequency)
Sxx_normed = (Sxx - Sxx.mean(axis=1)[:, None]) / Sxx.std(axis=1)[:, None]
vmax = abs(Sxx_normed).max()
plt.pcolormesh(t, f, Sxx_normed,
               cmap='RdBu_r', vmin=-vmax, vmax=vmax)
plt.colorbar()
plt.show()
