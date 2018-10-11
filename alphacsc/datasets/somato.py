import os
import mne
import numpy as np
from joblib import Memory
from scipy.signal import tukey

mem = Memory(location='.', verbose=0)


@mem.cache(ignore=['n_jobs'])
def load_data(sfreq=None, epoch=(-2, 4), n_jobs=1, filt=[2., None],
              n_splits=10, return_epochs=False, dataset='somato'):
    """Load and prepare the somato dataset for multiCSC


    Parameters
    ----------
    sfreq: float
        Sampling frequency of the signal. The data are resampled to match it.
    epoch : tuple or None
        If set to a tuple, extract epochs from the raw data, using
        t_min=epoch[0] and t_max=epoch[1]. Else, use the raw signal, divided
        in n_splits chunks.
    n_jobs : int
        Number of jobs that can be used for preparing (filtering) the data.
    filt : tuple of length 2
        Boundaries of filtering, e.g. (2, None), (30, 40), (None, 40).
    n_splits : int
        Number of splits to split the signal into.
    return_epochs : boolean
        If True, return epochs instead of X and info
    dataset : str in {'somato', 'sample'}
        Dataset to load.

    Returns
    -------
    X : array, shape (n_splits, n_channels, n_times)
        The loaded dataset.
    info : dict
        MNE dictionary of information about recording settings.
    """
    if dataset == 'somato':
        data_path = mne.datasets.somato.data_path()
        data_path = os.path.join(data_path, 'MEG', 'somato')
        file_name = os.path.join(data_path, 'sef_raw_sss.fif')
        raw = mne.io.read_raw_fif(file_name, preload=True)
        raw.notch_filter(np.arange(50, 101, 50), n_jobs=n_jobs)
        event_id = 1
    elif dataset == 'sample':
        data_path = mne.datasets.sample.data_path()
        data_path = os.path.join(data_path, 'MEG', 'sample')
        file_name = os.path.join(data_path, 'sample_audvis_raw.fif')
        raw = mne.io.read_raw_fif(file_name, preload=True)
        raw.notch_filter(np.arange(60, 181, 60), n_jobs=n_jobs)
        event_id = [1, 2, 3, 4]
    else:
        raise ValueError('Unknown parameter dataset=%s.' % (dataset, ))
    raw.filter(*filt, n_jobs=n_jobs)

    if epoch:
        t_min, t_max = epoch
        baseline = (None, 0)
        events = mne.find_events(raw, stim_channel='STI 014')
        events = mne.pick_events(events, include=event_id)

        picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                               stim=False)
        epochs = mne.Epochs(raw, events, event_id, t_min, t_max, picks=picks,
                            baseline=baseline, reject=dict(
                                grad=4000e-13, eog=350e-6), preload=True)
        epochs.pick_types(meg='grad', eog=False)
        if sfreq is not None:
            epochs = epochs.resample(sfreq, npad='auto', n_jobs=n_jobs)
        X = epochs.get_data()
        info = epochs.info
        if return_epochs:
            return epochs

    else:
        raw.pick_types(meg='grad', eog=False, stim=True)
        events = mne.find_events(raw, stim_channel='STI 014')
        events = mne.pick_events(events, include=event_id)
        raw.pick_types(meg='grad', stim=False)
        events[:, 0] -= raw.first_samp

        if sfreq is not None:
            raw, events = raw.resample(sfreq, events=events, npad='auto',
                                       n_jobs=n_jobs)

        X = raw.get_data()
        n_channels, n_times = X.shape
        n_times = n_times // n_splits
        X = X[:, :n_splits * n_times]
        X = X.reshape(n_channels, n_splits, n_times).swapaxes(0, 1)
        info = raw.info
        if return_epochs:
            raise ValueError('return_epochs=True is not allowed with '
                             'epochs=False')

    # XXX: causes problems when saving EvokedArray
    info['event_id'] = event_id
    info['events'] = events

    n_splits, n_channels, n_times = X.shape
    X *= tukey(n_times, alpha=0.1)[None, None, :]
    X /= np.std(X)
    return X, info
