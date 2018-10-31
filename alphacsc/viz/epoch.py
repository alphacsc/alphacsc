from copy import deepcopy

import mne
import numpy as np
from joblib import Parallel, delayed


def make_epochs(z_hat, info, t_lim, n_times_atom=1):
    """Make Epochs on the activations of atoms.
    n_splits, n_atoms, n_times_valid = z_hat.shape
    n_trials, n_atoms, n_times_epoch = z_hat_epoch.shape
    """
    n_splits, n_atoms, n_times_valid = z_hat.shape
    n_times = n_times_valid + n_times_atom - 1
    # pad with zeros
    padding = np.zeros((n_splits, n_atoms, n_times_atom - 1))
    z_hat = np.concatenate([z_hat, padding], axis=2)
    # reshape into an unique time-serie per atom
    z_hat = np.reshape(z_hat.swapaxes(0, 1), (n_atoms, n_splits * n_times))

    # create trials around the events, using mne
    new_info = mne.create_info(ch_names=n_atoms, sfreq=info['sfreq'])
    rawarray = mne.io.RawArray(data=z_hat, info=new_info, verbose=False)
    t_min, t_max = t_lim
    epochs = mne.Epochs(rawarray, info['events'], info['event_id'],
                        t_min, t_max, verbose=False)
    z_hat_epoched = epochs.get_data()
    return z_hat_epoched


def make_evoke(array, info, t_lim):
    """Compute evoked activations"""
    if array.ndim == 1:
        array = array[None, None]
    elif array.ndim == 2:
        array = array[None]
    epoched_array = make_epochs(array, info, t_lim=t_lim)
    evoked_array = epoched_array.mean(axis=0)
    return evoked_array


def make_evoke_one_surrogate(array, info, t_lim):
    # generate random events
    info = deepcopy(info)
    n_events = info['events'].shape[0]
    events = np.random.randint(array.shape[-1], size=n_events)
    events = np.sort(np.unique(events))
    n_events = events.shape[0]

    event_id = np.atleast_1d(info['event_id']).astype('int')
    n_tile = int(np.ceil(n_events / float(event_id.shape[0])))
    event_id_tiled = np.tile(event_id, n_tile)[:n_events]

    events = np.c_[events, np.zeros_like(events), event_id_tiled]
    info['events'] = events

    # make evoked with random events
    evoked_array = make_evoke(array, info, t_lim)
    return evoked_array


def make_evoke_all_surrogates(array, info, t_lim, n_jobs, n_surrogates=100):
    delayed_func = delayed(make_evoke_one_surrogate)
    evoked_arrays = Parallel(n_jobs=n_jobs)(delayed_func(array, info, t_lim)
                                            for i in range(n_surrogates))
    return np.array(evoked_arrays)
