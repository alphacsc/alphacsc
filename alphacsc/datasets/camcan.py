""" Cambridge Centre for Ageing and Neuroscience (CamCAN) dataset
Data available at: https://www.cam-can.org/index.php?content=dataset

Shafto, M.A., Tyler, L.K., Dixon, M., Taylor, J.R., Rowe, J.B., Cusack, R.,
Calder, A.J., Marslen-Wilson, W.D., Duncan, J., Dalgleish, T., Henson, R.N.,
Brayne, C., CamCAN, & Matthews, F.E. (2014). The Cambridge Centre for Ageing
and Neuroscience (CamCAN) study protocol: a cross-sectional, lifespan,
multidisciplinary examination of healthy cognitive ageing. BMC Neurology,
14(204). doi:10.1186/s12883-014-0204-1
https://bmcneurol.biomedcentral.com/articles/10.1186/s12883-014-0204-1 
"""
import os
from os.path import join
from copy import deepcopy

import mne
import numpy as np
import pandas as pd
from joblib import Memory

from mne_bids import BIDSPath, read_raw_bids

try:
    from ..utils.signal import split_signal
except (ValueError, ImportError):
    from alphacsc.utils.signal import split_signal


mem = Memory(location='.', verbose=0)


def get_subject_info(subject_id, participants_file):
    """Get the subject's informations

    Parameters
    ----------
    subject_id : str
        Subject id, similar to the name of its corresponding folder
    participants_file : str
        Path to the CamCAN subjects' informations file.

    Returns
    -------
    subject_info : dict
        keys are 'participant_id', 'age', 'sex', 'hand'
    """

    participants = pd.read_csv(participants_file, sep='\t')
    participants.drop(participants.columns[-1], axis=1, inplace=True)
    subject_info = participants[participants['participant_id'] == subject_id]\
        .iloc[0]\
        .to_dict()

    return subject_info


@mem.cache(ignore=['n_jobs'])
def load_data(BIDS_root, sss_cal, ct_sparse, subject_id='sub-CC110033',
              n_splits=10, sfreq=None, epoch=None, filter_params=[2., 45],
              return_array=True, n_jobs=1):
    """Load and prepare the CamCAN dataset of one subject for multiCSC

    Parameters
    ----------
    BIDS_root : str
        The root directory of the BIDS dataset.
    sss_cal, ct_sparse : str
        Path to the calibration file and cross-talk file for Maxwell filter
    subject_id : str
        Subject id, similar to the name of its corresponding folder
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
    # path to the CamCAN subjects' informations file
    participants_file = join(BIDS_root, "participants.tsv")

    pick_types_epoch = dict(meg='grad', eeg=False, eog=True, stim=False)
    pick_types_final = dict(meg='grad', eeg=False, eog=False, stim=False)

    # print subject's age and sex information
    subject_info = get_subject_info(subject_id, participants_file)
    age, sex = subject_info['age'], subject_info['sex']
    print(f"Subject {subject_id.split('-')[1]}: {age} year-old {sex}")

    bp = BIDSPath(
        root=BIDS_root,
        subject=subject_id.split('-')[1],
        task="smt",
        datatype="meg",
        extension=".fif",
        session="smt",
    )
    raw = read_raw_bids(bp)
    raw.load_data()
    raw.notch_filter([50, 100])
    raw = mne.preprocessing.maxwell_filter(
        raw,
        calibration=sss_cal,
        cross_talk=ct_sparse,
        st_duration=10.0
    )

    event_id = {
        'audiovis/1200Hz': 1,  # bimodal
        'audiovis/300Hz': 2,   # bimodal
        'audiovis/600Hz': 3,   # bimodal
        'button': 4,           # button press
        'catch/0': 5,          # unimodal auditory
        'catch/1': 6           # unimodal visual
    }
    events, _ = mne.events_from_annotations(raw)
    events = mne.pick_events(events, include=list(event_id.values()))

    raw.filter(*filter_params, n_jobs=n_jobs)

    if epoch:
        t_min, t_max = epoch
        baseline = (None, 0)

        picks = mne.pick_types(raw.info, **pick_types_epoch)
        epochs = mne.Epochs(
            raw, events, event_id, t_min, t_max, picks=picks,
            baseline=baseline, reject=dict(grad=4000e-13, eog=350e-6),
            preload=True
        )
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
            # recompute n_jobs and n_splits so it is optimal by ensuring
            # n_splits is a multiple of n_jobs
            n_jobs = min(n_jobs, os.cpu_count())
            k = n_splits // n_jobs
            n_splits = min(n_splits, n_jobs * k)
            # split X
            X = raw.get_data()
            X = split_signal(X, n_splits=n_splits, apply_window=True)

    # Deep copy before modifying info to avoid issues when saving EvokedArray
    info = deepcopy(info)
    event_info = dict(
        event_id=event_id, events=events, subject_info=subject_info)

    info['temp'] = event_info

    if return_array:
        X /= np.std(X)
        return X, info
    elif epoch:
        return epoch, info
    else:
        return raw, info
