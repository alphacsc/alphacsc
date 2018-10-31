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
    epochs = mne.Epochs(rawarray, info['events'], info['event_id'], t_min,
                        t_max, verbose=False)
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


def plot_evoked_surrogates(array, info, t_lim, ax, n_jobs, label='',
                           threshold=0.005):
    """Compute and plot evoked array distribution over random events"""
    assert array.ndim == 1
    assert ax is not None
    # compute mean over epochs
    evoked = make_evoke(array, info, t_lim)[0]

    # compute surrogate evoked with random events
    evoked_surrogate = make_evoke_all_surrogates(array, info, t_lim,
                                                 n_jobs)[:, 0]

    # find thresholds
    low, high = 100 * threshold / 2., 100 * (1 - threshold / 2.)
    threshold_low = np.percentile(evoked_surrogate.min(axis=1), low)
    threshold_high = np.percentile(evoked_surrogate.max(axis=1), high)

    # plot the evoked and a gray area for the 95% percentile
    t = np.arange(len(evoked)) / info['sfreq'] + t_lim[0]
    outside_thresholds = ((evoked > threshold_high) + (evoked < threshold_low))
    color = 'C1' if np.any(outside_thresholds) else 'C2'
    ax.plot(t, evoked, label=label, color=color)
    label_th = str(100 * (1 - threshold)) + ' %'
    ax.fill_between(t, threshold_low, threshold_high, color='k', alpha=0.2,
                    label=label_th)
    ax.fill_between(t, threshold_low, threshold_high, where=outside_thresholds,
                    color='y', alpha=0.2)
    ax.axvline(0, color='k', linestyle='--')
    ax.set_ylim([0, None])
    ax.legend()

    # # plot the histogram of evoked_surrogate, and of evoked
    # ax = axes[1]
    # ax.hist(evoked_surrogate, bins=100, density=True, label='surrogate')
    # ax.hist(evoked.ravel(), bins=100, density=True, alpha=0.8,
    #         label='evoked')
    # ax.axvline(threshold_low, color='k', linestyle='--')
    # ax.axvline(threshold_high, color='k', linestyle='--')
    # ax.legend()
