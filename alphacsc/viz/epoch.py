from copy import deepcopy

import mne
import numpy as np
from joblib import Parallel, delayed

from ..utils.convolution import construct_sources


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
    info_temp = info['temp']
    epochs = mne.Epochs(rawarray, info_temp['events'], info_temp['event_id'],
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
    info_temp = info['temp']
    n_events = info_temp['events'].shape[0]
    events = np.random.randint(array.shape[-1], size=n_events)
    events = np.sort(np.unique(events))
    n_events = events.shape[0]

    event_id = np.atleast_1d(info_temp['event_id']).astype('int')
    n_tile = int(np.ceil(n_events / float(event_id.shape[0])))
    event_id_tiled = np.tile(event_id, n_tile)[:n_events]

    events = np.c_[events, np.zeros_like(events), event_id_tiled]
    info_temp['events'] = events

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


def plot_epochs_of_selected_atoms(model, z_hat, X_full, info, t_lim,
                                  idx_atoms, channel=None,
                                  align=False):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    topomap = model.u_hat_
    v_hat = model.v_hat_
    sources = construct_sources(model, z_hat)

    color_atoms = ['C%d' % d for d in [1, 2, 4, 5, 6, 7, 8, 9]]

    # check idx_atoms
    if isinstance(idx_atoms, str) and idx_atoms == 'all':
        idx_atoms = np.arange(topomap.shape[0])
    idx_atoms = np.atleast_1d(idx_atoms)
    n_atoms = len(idx_atoms)

    # select one channel
    if channel is None:
        u_ks = topomap[idx_atoms]
        channel = np.argsort(np.sum(np.abs(u_ks), axis=0))[-1]
    trut = X_full[channel]

    # select atoms
    pred = np.dot(topomap[idx_atoms].T, sources[idx_atoms])
    pred = pred[channel]
    acti = z_hat[0, idx_atoms]
    v_hat = v_hat[idx_atoms]
    # roll to put activation in the peak of the atoms
    for kk in range(n_atoms):
        shift = np.argmax(np.abs(v_hat[kk]))
        acti[kk] = np.roll(acti[kk], shift)
    epoched_acti = make_epochs(acti[None], info, t_lim=t_lim)

    epoched_trut = make_epochs(trut[None, None], info, t_lim=t_lim)[:, 0]
    epoched_pred = make_epochs(pred[None, None], info, t_lim=t_lim)[:, 0]
    n_epochs, n_times_epoch = epoched_trut.shape
    time_array = np.linspace(*t_lim, n_times_epoch)
    vmax = max(
        np.percentile(np.abs(epoched_trut), 95),
        np.percentile(np.abs(epoched_pred), 95))

    # prepare plots
    n_ids = len(info['event_id'])
    nrows = 4
    height = 11
    fig, axes = plt.subplots(nrows, n_ids, figsize=(4 * n_ids, height),
                             sharex=True)

    for jj, this_event_id in enumerate(info['event_id']):
        mask = info['events'][:, -1] == this_event_id
        masked_trut = epoched_trut[mask, :].copy()
        masked_pred = epoched_pred[mask, :].copy()
        n_epochs_event = masked_pred.shape[0]

        # order epochs
        if np.mean(masked_pred) > 0:
            order = np.argsort(np.argmax(masked_pred, axis=1))
        else:
            order = np.argsort(np.argmin(masked_pred, axis=1))
        masked_trut = masked_trut[order]
        masked_pred = masked_pred[order]
        masked_acti = epoched_acti[mask].copy()[order]

        onsets = np.argmin(np.abs(time_array)) * np.ones(
            n_epochs_event, dtype='int')

        # experimental: roll to align activations
        if align:
            roll_idx = np.argmin(masked_pred, axis=1)
            mask = roll_idx != 0
            roll_idx[mask] -= np.int_(np.median(roll_idx[mask]))
            roll_idx = [idx if -20 < idx < 20 else 0 for idx in roll_idx]
            masked_trut = np.array([
                np.roll(epoch, -idx)
                for epoch, idx in zip(masked_trut, roll_idx)
            ])
            masked_pred = np.array([
                np.roll(epoch, -idx)
                for epoch, idx in zip(masked_pred, roll_idx)
            ])
            onsets += roll_idx

        # for the channel data, and the channel data as approximated by 1 atom
        for ii, epoched in enumerate([masked_trut, masked_pred]):
            atom_str = '%d-atom%s' % (n_atoms, 's' if n_atoms > 1 else '')
            label = ['raw', atom_str][ii]
            color = ['C0', 'C3'][ii]

            # first plot the epochs
            ax = axes[2 * ii, jj]
            extent = (*t_lim, 0, n_epochs_event)
            ax.imshow(epoched, cmap='RdBu_r', aspect='auto', vmin=-vmax,
                      vmax=vmax, extent=extent, interpolation='bilinear',
                      origin='lower')
            ax.set(title='Event %d - %s' % (this_event_id, label))
            ax.set(xlabel='Time (s)', ylabel='Epoch')
            ax.title.set_color(color)

            # plot the event onsets in black
            ax = axes[2 * ii, jj]

            ax.plot(time_array[onsets], np.arange(n_epochs_event), 'k')

            # plot the activations instants
            if ii == 1:
                ax = axes[2 * ii, jj]
                for kk in range(n_atoms):
                    xx, yy = masked_acti[:, kk, :].nonzero()
                    values = masked_acti[xx, kk, yy]
                    values /= values.max()
                    color_array = np.asarray([(*colors.to_rgb(color_atoms[kk]),
                                               alpha) for alpha in values])
                    ax.scatter(time_array[yy], xx, marker='.',
                               color=color_array)
                    ax.set_ylim(0, n_epochs_event)

            # then plot the mean evoked response
            ax = axes[1, jj]
            ax.get_shared_y_axes().join(ax, axes[1, 0])
            mean, std = epoched.mean(axis=0), epoched.std(axis=0) / 2.
            ax.plot(time_array, mean, label=label, color=color)
            ax.fill_between(time_array, mean - std, mean + std, color=color,
                            alpha=0.3)
            ax.axvline(0, color='k')
            ax.set(xlabel='Time (s)', title='Evoked')
            ax.legend()
            ax.grid(True)

            # plot the activations distributions
            if ii == 1:
                ax = axes[3, jj]
                ax.get_shared_y_axes().join(ax, axes[3, 0])
                for kk in range(n_atoms):
                    xx, yy = masked_acti[:, kk, :].nonzero()
                    ax.hist(time_array[yy], bins=n_times_epoch, range=t_lim,
                            label='atom %d' % idx_atoms[kk], alpha=0.6,
                            color=color_atoms[kk])
                ax.set(xlabel='Time (s)', title='Activations')
                ax.legend()
    ax.set_xlim(t_lim)
    plt.tight_layout()
