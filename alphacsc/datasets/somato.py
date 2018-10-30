from os.path import join
from copy import deepcopy

import mne
import numpy as np
from joblib import Memory
from scipy.signal import tukey

mem = Memory(location='.', verbose=0)


@mem.cache(ignore=['n_jobs'])
def load_data(dataset="somato", n_splits=10, sfreq=None, epoch=None,
              filter_params=[2., None], return_array=True, n_jobs=1):
    """Load and prepare the somato dataset for multiCSC

    Parameters
    ----------
    dataset : str in {'somato', 'sample'}
        Dataset to load.
    n_splits : int
        Split the signal in n_split signals of same length before returning it.
        If epoch is provided, the signal is instead splitted according to the
        epochs and this option is not followed.
    sfreq : float
        Sampling frequency of the signal. The data are resampled to match it.
    epoch : tuple or None
        If set to a tuple, extract epochs from the raw data, using
        t_min=epoch[0] and t_max=epoch[1]. Else, use the raw signal, divided
        in n_splits chunks.
    filter_params : tuple of length 2
        Boundaries of filtering, e.g. (2, None), (30, 40), (None, 40).
    return_array : boolean
        If True, return an NumPy array, instead of mne objects.
    n_jobs : int
        Number of jobs that can be used for preparing (filtering) the data.

    Returns
    -------
    X : array, shape (n_splits, n_channels, n_times)
        The loaded dataset.
    info : dict
        MNE dictionary of information about recording settings.
    """

    pick_types_epoch = dict(meg='grad', eeg=False, eog=True, stim=False)
    pick_types_final = dict(meg='grad', eeg=False, eog=False, stim=False)

    if dataset == 'somato':
        data_path = mne.datasets.somato.data_path()
        subjects_dir = join(data_path, "subjects")
        data_dir = join(data_path, 'MEG', 'somato')
        file_name = join(data_dir, 'sef_raw_sss.fif')
        raw = mne.io.read_raw_fif(file_name, preload=True)
        raw.notch_filter(np.arange(50, 101, 50), n_jobs=n_jobs)
        event_id = 1

        # Dipole fit information
        cov = None  # see below
        file_trans = join(data_dir, "sef_raw_sss-trans.fif")
        file_bem = join(subjects_dir, 'somato', 'bem',
                        'somato-5120-bem-sol.fif')

    elif dataset == 'sample':
        data_path = mne.datasets.sample.data_path()
        subjects_dir = join(data_path, "subjects")
        data_dir = join(data_path, 'MEG', 'sample')
        file_name = join(data_dir, 'sample_audvis_raw.fif')
        raw = mne.io.read_raw_fif(file_name, preload=True)
        raw.notch_filter(np.arange(60, 181, 60), n_jobs=n_jobs)
        event_id = [1, 2, 3, 4]

        # Dipole fit information
        cov = join(data_dir, 'sample_audvis-cov.fif')
        file_trans = join(data_dir, 'sample_audvis_raw-trans.fif')
        file_bem = join(subjects_dir, 'sample', 'bem',
                        'sample-5120-bem-sol.fif')

    else:
        raise ValueError('Unknown parameter dataset=%s.' % (dataset, ))
    raw.filter(*filter_params, n_jobs=n_jobs)

    baseline = (None, 0)
    events = mne.find_events(raw, stim_channel='STI 014')
    events = mne.pick_events(events, include=event_id)

    # compute the covariance matrix for somato
    if dataset == "somato":
        picks_cov = mne.pick_types(raw.info, **pick_types_epoch)
        epochs_cov = mne.Epochs(raw, events, event_id, tmin=-4, tmax=0,
                                picks=picks_cov, baseline=baseline,
                                reject=dict(grad=4000e-13, eog=350e-6),
                                preload=True)
        epochs_cov.pick_types(**pick_types_final)
        cov = mne.compute_covariance(epochs_cov)

    if epoch:
        t_min, t_max = epoch

        picks = mne.pick_types(raw.info, **pick_types_epoch)
        epochs = mne.Epochs(raw, events, event_id, t_min, t_max, picks=picks,
                            baseline=baseline, reject=dict(
                                grad=4000e-13, eog=350e-6), preload=True)
        epochs.pick_types(**pick_types_final)
        info = epochs.info
        if sfreq is not None:
            epochs = epochs.resample(sfreq, npad='auto', n_jobs=n_jobs)

        if return_array:
            X = epochs.get_data()

    else:
        events[:, 0] -= raw.first_samp
        raw.pick_types(**pick_types_final)
        info = raw.info

        if sfreq is not None:
            raw, events = raw.resample(sfreq, events=events, npad='auto',
                                       n_jobs=n_jobs)

        if return_array:
            X = raw.get_data()
            n_channels, n_times = X.shape
            n_times = n_times // n_splits
            X = X[:, :n_splits * n_times]
            X = X.reshape(n_channels, n_splits, n_times).swapaxes(0, 1)

    # Deep copy before modifying info to avoid issues when saving EvokedArray
    info = deepcopy(info)
    info['events'] = events
    info['event_id'] = event_id
    info['subject'] = dataset
    info['subjects_dir'] = subjects_dir

    info['cov'] = cov
    info['file_bem'] = file_bem
    info['file_trans'] = file_trans

    if return_array:
        n_splits, n_channels, n_times = X.shape
        X *= tukey(n_times, alpha=0.1)[None, None, :]
        X /= np.std(X)
        return X, info
    elif epoch:
        return epoch, info
    else:
        return raw, info
