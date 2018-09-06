import mne
import numpy as np


def make_epochs(z_hat, info, n_times_atom):
    """Make Epochs on the activations of atoms.
    n_atoms, n_splits, n_times_valid = z_hat.shape
    n_atoms, n_trials, n_times_epoch = z_hat_epoched.shape
    """
    n_atoms, n_splits, n_times_valid = z_hat.shape
    n_times = n_times_valid + n_times_atom - 1
    # pad with zeros
    padding = np.zeros((n_atoms, n_splits, n_times_atom - 1))
    z_hat = np.concatenate([z_hat, padding], axis=2)
    # reshape into an unique time-serie per atom
    z_hat = np.reshape(z_hat, (n_atoms, n_splits * n_times))

    # create trials around the events, using mne
    new_info = mne.create_info(ch_names=n_atoms, sfreq=info['sfreq'])
    rawarray = mne.io.RawArray(data=z_hat, info=new_info, verbose=False)
    tmin, tmax = -2., 5.
    epochs = mne.Epochs(rawarray, info['events'], info['event_id'], tmin, tmax,
                        verbose=False)
    z_hat_epoched = np.swapaxes(epochs.get_data(), axis1=0, axis2=1)
    return z_hat_epoched
