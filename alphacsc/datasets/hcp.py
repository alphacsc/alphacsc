import re
import os
import hcp
import mne
import numpy as np
from glob import glob
from copy import deepcopy
from joblib import Memory
from hcp.io.file_mapping.file_mapping import kind_map

from ..utils import check_random_state


HCP_DIR = "/storage/store/data/HCP900/"
CONVERSION_MAP = {v: k for k, v in kind_map.items()}


mem = Memory(location='.', verbose=0)


def get_all_records(hcp_path=HCP_DIR):
    """Make a dictionary with all HCP files in the directory hcp_path

    Parameters
    ----------
    hcp_path: str
        Path in which the HCP files are located

    Return
    ------
    db: dict
        Dictionary with {data_type: {subject: [run_index_0, run_index_1, ...]}}
        The keys are str for the type of exercises and the values are
        dictionaries containing a list per subject with the run_indexes.

    """
    # List all files with unprocesses
    pattern = os.path.join(hcp_path, "*/unprocessed/MEG/*/4D/config")
    list_files = glob(pattern)
    db = {}
    pattern = pattern.replace("*", "(.*)")
    pattern = pattern.replace("MEG/", "MEG/\d+-")  # noqa
    for f_name in list_files:
        subject, data_type = re.match(pattern, f_name).groups()
        data_type = CONVERSION_MAP[data_type]
        type_subjects = db.get(data_type, {})
        type_subject_records = type_subjects.get(subject, [])
        type_subject_records += [len(type_subject_records)]
        type_subjects[subject] = type_subject_records
        db[data_type] = type_subjects
    print("Found {} types".format(len(db.keys())))

    return db


@mem.cache(ignore=['n_jobs'])
def load_one_record(data_type, subject, run_index, sfreq=300, epoch=None,
                    filter_params=[5., None], n_jobs=1):
    # Load the record and correct the sensor space to get proper visualization
    print(f"subject={subject}, data_type={data_type}, run_index={run_index}, "
          f"hcp_path={HCP_DIR}")
    raw = hcp.read_raw(subject, data_type=data_type, run_index=run_index,
                       hcp_path=HCP_DIR, verbose=0)
    raw.load_data()
    hcp.preprocessing.map_ch_coords_to_mne(raw)
    raw.pick_types(meg='mag', eog=False, stim=True)

    # filter the electrical and low frequency components
    raw.notch_filter([60, 120], n_jobs=n_jobs)
    raw.filter(*filter_params, n_jobs=n_jobs)

    # Resample to the requested sfreq
    if sfreq is not None:
        raw.resample(sfreq=sfreq, n_jobs=n_jobs)

    events = mne.find_events(raw)
    raw.pick_types(meg='mag', stim=False)
    events[:, 0] -= raw.first_samp

    # Deep copy before modifying info to avoid issues when saving EvokedArray
    info = deepcopy(raw.info)
    info['events'] = events
    info['event_id'] = np.unique(events[:, 2])

    # Return the data
    return raw.get_data(), info


def load_data(n_trials=10, data_type='rest', sfreq=150, epoch=None,
              filter_params=[5., None], equalize="zeropad", n_jobs=1,
              random_state=None):
    """Load and prepare the HCP dataset for multiCSC


    Parameters
    ----------
    n_trials : int
        Number of recordings that are loaded.
    data_type : str
        Type of recordings loaded. Should be in {'rest', 'task_working_memory',
        'task_motor', 'task_story_math', 'noise_empty_room', 'noise_subject'}.
    sfreq : float
        Sampling frequency of the signal. The data are resampled to match it.
    epoch : tuple or None
        If set to a tuple, extract epochs from the raw data, using
        t_min=epoch[0] and t_max=epoch[1]. Else, use the raw signal, divided
        in n_splits chunks.
    filter_params : tuple
        Frequency cut for a band pass filter applied to the signals. The
        default is a high-pass filter with frequency cut at 2Hz.
    n_jobs : int
        Number of jobs that can be used for preparing (filtering) the data.
    random_state : int | None
        State to seed the random number generator.

    Return
    ------
    X : ndarray, shape (n_trials, n_channels, n_times)
        Signals loaded from HCP.
    info : list of mne.Info
        List of the info related to each signals.
    """
    if data_type == "rest" and epoch is not None:
        raise ValueError("epoch != None is not valid with resting-state data.")

    rng = check_random_state(random_state)
    mne.set_log_level(30)

    db = get_all_records()
    records = [(subject, run_index)
               for subject, runs in db[data_type].items()
               for run_index in runs]

    X, info = [], []
    records = rng.permutation(records)[:n_trials]
    for i, (subject, run_index) in enumerate(records):
        print("\rLoading HCP subjects: {:7.2%}".format(i / n_trials),
              end='', flush=True)
        X_n, info_n = load_one_record(
            data_type, subject, int(run_index), sfreq=sfreq, epoch=epoch,
            filter_params=filter_params, n_jobs=n_jobs)
        X += [X_n]
        info += [info_n]

    print("\rLoading HCP subjects: done   ")
    X = make_array(X, equalize=equalize)
    X /= np.std(X)
    return X, info


def data_generator(n_trials=10, data_type='rest', sfreq=150, epoch=None,
                   filter_params=[5., None], equalize="zeropad", n_jobs=1,
                   random_state=None):
    """Generator loading subjects from the HCP dataset for multiCSC


    Parameters
    ----------
    n_trials : int
        Number of recordings that are loaded.
    data_type : str
        Type of recordings loaded. Should be in {'rest', 'task_working_memory',
        'task_motor', 'task_story_math', 'noise_empty_room', 'noise_subject'}.
    sfreq : float
        Sampling frequency of the signal. The data are resampled to match it.
    epoch : tuple or None
        If set to a tuple, extract epochs from the raw data, using
        t_min=epoch[0] and t_max=epoch[1]. Else, use the raw signal, divided
        in n_splits chunks.
    filter_params : tuple
        Frequency cut for a band pass filter applied to the signals. The
        default is a high-pass filter with frequency cut at 2Hz.
    n_jobs : int
        Number of jobs that can be used for preparing (filtering) the data.
    random_state : int | None
        State to seed the random number generator.

    Yields
    ------
    X : ndarray, shape (1, n_channels, n_times)
        Signals loaded from HCP.
    info : list of mne.Info
        info related to this signal.
    """
    if data_type == "rest" and epoch is not None:
        raise ValueError("epoch != None is not valid with resting-state data.")

    rng = check_random_state(random_state)
    mne.set_log_level(30)

    db = get_all_records()
    records = [(subject, run_index)
               for subject, runs in db[data_type].items()
               for run_index in runs]

    records = rng.permutation(records)[:n_trials]
    for i, (subject, run_index) in enumerate(records):
        try:
            X_n, info_n = load_one_record(
                data_type, subject, int(run_index), sfreq=sfreq, epoch=epoch,
                filter_params=filter_params, n_jobs=n_jobs)
            X_n /= X_n.std()
            yield X_n, info_n
        except UnicodeDecodeError:
            print("failed to load {}-{}-{}"
                  .format(subject, data_type, run_index))


def make_array(X, equalize='zeropad'):
    """"""
    x_shape = np.array([x.shape for x in X])
    assert np.all(x_shape[..., :-1] == x_shape[0, ..., :-1])
    if equalize == "crop":
        L = x_shape.min(axis=0)[-1]
        X = np.array([x[..., :L] for x in X])
    elif equalize == "zeropad":
        X_shape = tuple(x_shape.max(axis=0))
        X_shape, L = X_shape[:-1], X_shape[-1]
        X = np.array([
            np.concatenate([x, np.zeros(X_shape + (L - x.shape[-1], ))],
                           axis=-1) for x in X
        ])
    else:
        raise ValueError("The equalize '{}' is not valid. It should be in "
                         "{'crop', 'zeropad'}".format(equalize))

    return X
