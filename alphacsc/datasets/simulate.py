import os
import mne
import numpy as np
from joblib import Memory
from scipy.signal import tukey

try:
    from ..utils import check_random_state
except ValueError:
    from alphacsc.utils import check_random_state


mem = Memory(location='.', verbose=0)


@mem.cache()
def load_data(n_trials=40, n_channels=1, n_times=6, sigma=.05, sfreq=300,
              f_noise=True, random_state=None, n_jobs=4):
    """Simulate data with 10Hz mu-wave and 10Hz oscillations.

    Parameters
    ----------
    n_trials : int
        Number of simulated signals
    n_channels : int
        Number of channels in the signals
    n_times : float
        Length of the generated signals, in seconds. The generated signal
        will have length n_times * sfreq
    sigma : float
        Additive noise level in the signal
    sfreq : int
        Sampling rate for the signal
    f_noise : boolean
        If set to True, use MNE empty room data as a noise in the signal
    n_jobs : int
        Number of processes used to filter the f_noise
    random_state : int | None
        State to seed the random number generator

    Return
    ------
    X : ndarray, shape (n_trials, n_channels, n_times)
        Simulated 10Hz sinusoidal signals with 10Hz mu-wave between 2sec and
        [3, 3.5]sec. some random phases are applied to the different part of
        the signal.
    info : dict
        Contains the topomap 'u' associated to each component of the signal.
    """
    rng = check_random_state(random_state)

    freq = 10  # Generate 10Hz mu-wave
    phase_shift = rng.rand(n_trials, 1) * sfreq * np.pi
    t0 = 2
    L = 1. + .5 * rng.rand(n_trials, 1)
    t = (np.linspace(0., n_times, int(n_times * sfreq)))
    mask = (t[None] > t0) * (t[None] < t0 + L)
    t *= 2 * np.pi * freq
    t = t[None] + phase_shift
    noisy_phase = .5 * np.sin(t / (3 * np.sqrt(2)))
    phi_t = t + noisy_phase
    signal = np.sin(phi_t + np.cos(phi_t) * mask)

    U = rng.randn(1, n_channels, 1)
    U_mu = rng.randn(1, n_channels, 1)
    U /= np.sqrt((U * U).sum()) * np.sign(U.sum())
    U_mu /= np.sqrt((U_mu * U_mu).sum()) * np.sign(U_mu.sum())
    signal = (U + (U_mu - U) * mask[:, None]) * signal[:, None]

    # signal += sigma * rng.randn(*signal.shape)

    # generate noise from empty room noise inthe mne.sample dataset
    if f_noise:
        data_path = os.path.join(mne.datasets.sample.data_path(), 'MEG',
                                 'sample')
        raw = mne.io.read_raw_fif(os.path.join(data_path, 'ernoise_raw.fif'),
                                  preload=True)

        raw.pick_types(meg='mag')
        nyquist = raw.info['sfreq'] / 2.
        raw.notch_filter(np.arange(60, nyquist - 10., 60), n_jobs=n_jobs)
        raw.filter(.5, None, n_jobs=n_jobs)
        X = raw[:][0]
        X /= X.std(axis=1, keepdims=True)
        max_channels, T_max = X.shape

        channels = rng.choice(max_channels, n_channels)
        L_sig = int(n_times * sfreq)
        for i in range(n_trials):
            t = rng.choice(T_max - L_sig)
            signal[i] += sigma * X[channels, t:t + L_sig]
    else:
        signal += sigma * rng.randn(signal.shape)

    signal *= tukey(signal.shape[-1], alpha=0.05)[None, None, :]

    info = {}
    info['u'] = np.r_[U[:, :, 0], U_mu[:, :, 0]]

    return signal, info


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    load_data.clear()
    X, info = load_data(n_trials=1000, sigma=1, n_channels=1)
    plt.plot(np.mean(X, axis=0).T)
    plt.plot(X[0].T)
    plt.show()
