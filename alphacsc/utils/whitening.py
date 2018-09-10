import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from .arma import Arma


def whitening(X, ordar=10, block_length=256, sfreq=1., zero_phase=True,
              plot=False, use_fooof=False):
    n_trials, n_channels, n_times = X.shape

    ar_model = Arma(ordar=ordar, ordma=0, fs=sfreq, block_length=block_length)
    ar_model.periodogram(X.reshape(-1, n_times), hold=False, mean_psd=True)

    if use_fooof:  # pragma: no cover
        # Fit the psd with a 1/f^a background model plus a gaussian mixture.
        # We keep only the background model
        # (pip install fooof)
        from fooof import FOOOF
        fm = FOOOF(background_mode='fixed', verbose=False)
        power_spectrum = ar_model.psd[-1][0]
        freqs = np.linspace(0, sfreq / 2.0, len(power_spectrum))

        fm.fit(freqs, power_spectrum, freq_range=None)
        # repete first point, which is f_0
        bg_fit = np.r_[fm._bg_fit[0], fm._bg_fit][None, :]
        ar_model.psd.append(np.power(10, bg_fit))

    if zero_phase:
        ar_model.psd[-1] = np.sqrt(ar_model.psd[-1])
    ar_model.estimate()

    # apply the whitening for zero-phase filtering
    X_white = apply_whitening(ar_model, X, zero_phase=zero_phase, mode='same')
    assert X_white.shape == X.shape

    # removes edges
    n_times_white = X_white.shape[-1]
    X_white *= signal.tukey(n_times_white,
                            alpha=3 / float(n_times_white))[None, None, :]

    if plot:  # pragma: no cover
        # plot the Power Spectral Density (PSD) before/after
        ar_model.arma2psd(hold=True)
        if zero_phase:
            ar_model.psd[-2] **= 2
            ar_model.psd[-1] **= 2
        ar_model.periodogram(
            X_white.reshape(-1, n_times), hold=True, mean_psd=True)
        labels = ['signal', 'model AR', 'signal white']
        if use_fooof:
            labels = ['signal', 'FOOOF fit', 'model AR', 'signal white']
        ar_model.plot('periodogram before/after whitening',
                      labels=labels, fscale='lin')
        plt.legend(loc='lower left')

    return ar_model, X_white


def apply_ar(ar_model, x, zero_phase=True, mode='same', reverse_ar=False):
    # TODO: speed-up with only one conv
    ar_coef = np.concatenate((np.ones(1), ar_model.AR_))
    if reverse_ar:
        ar_coef = ar_coef[::-1]

    if zero_phase:
        tmp = signal.fftconvolve(x, ar_coef, mode)[::-1]
        return signal.fftconvolve(tmp, ar_coef, mode)[::-1]
    else:
        return signal.fftconvolve(x, ar_coef, mode)


def apply_whitening(ar_model, X, zero_phase=True, mode='same',
                    reverse_ar=False, n_channels=None):
    if X.ndim == 2:
        msg = "For rank1 D, n_channels should be provided"
        assert n_channels is not None, msg
        v = X[:, n_channels:]
        v_white = [apply_ar(ar_model, vk, zero_phase=zero_phase,
                            mode=mode) for vk in v]
        return np.c_[X[:, :n_channels], v_white]

    elif X.ndim == 3:
        return np.array([[
            apply_ar(ar_model, Xij, zero_phase=zero_phase, mode=mode,
                     reverse_ar=reverse_ar) for Xij in Xi]
            for Xi in X])
    else:
        raise NotImplementedError("Should not be called!")


def unwhitening(ar_model, X_white, estimate=True, zero_phase=True, plot=False):
    if X_white.ndim == 2:
        X_white = X_white[None, :]

    n_trials, n_channels, n_times = X_white.shape

    if estimate:
        ar_model.arma2psd(hold=True)
        ar_model.psd[-1] = 1. / ar_model.psd[-1]

        ar_model.estimate()

    # apply the whitening twice (forward and backward) for zero-phase filtering
    X_unwhite = apply_whitening(ar_model, X_white, zero_phase=zero_phase)
    assert X_unwhite.shape == X_white.shape

    if plot:
        # plot the Power Spectral Density (PSD) before/after
        ar_model.periodogram(
            X_white.reshape(-1, n_times), hold=False, mean_psd=True)
        ar_model.arma2psd(hold=True)
        if zero_phase:
            ar_model.psd[-1] **= 2
        ar_model.periodogram(
            X_unwhite.reshape(-1, n_times), hold=True, mean_psd=True)
        ar_model.plot('periodogram before/after unwhitening',
                      labels=['signal white', 'model AR',
                              'signal unwhite'], fscale='lin')
        plt.legend(loc='lower left')
    return X_unwhite
