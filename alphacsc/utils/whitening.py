import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tukey

# pip install git+https://github.com/pactools/pactools.git#egg=pactools
from pactools.utils.arma import Arma
from pactools.utils.pink_noise import almost_pink_noise


def whitening(X, ordar=10, block_length=256, sfreq=1., zero_phase=True,
              plot=False):
    n_trials, n_channels, n_times = X.shape

    ar_model = Arma(ordar=ordar, ordma=0, fs=sfreq, block_length=block_length)
    ar_model.periodogram(X.reshape(-1, n_times), hold=False, mean_psd=True)

    if zero_phase:
        ar_model.psd[-1] = np.sqrt(ar_model.psd[-1])
    ar_model.estimate()

    # apply the whitening twice (forward and backward) for zero-phase filtering
    if zero_phase:
        X_white = [[
            ar_model.inverse(ar_model.inverse(X_np)[::-1])[::-1]
            for X_np in X_n
        ] for X_n in X]
    else:
        X_white = [[ar_model.inverse(X_np) for X_np in X_n] for X_n in X]

    X_white = np.array(X_white)
    assert X_white.shape == X.shape

    # removes edges
    X_white *= tukey(n_times, alpha=3 / float(n_times))[None, None, :]

    if plot:
        # plot the Power Spectral Density (PSD) before/after
        ar_model.arma2psd(hold=True)
        ar_model.periodogram(
            X_white.reshape(-1, n_times), hold=True, mean_psd=True)
        ar_model.plot('periodogram before/after whitening',
                      labels=['signal', 'model AR',
                              'signal white'], fscale='lin')
        plt.legend(loc='lower left')

    return ar_model, X_white


def unwhitening(ar_model, X_white, estimate=True, zero_phase=True, plot=False):
    if X_white.ndim == 2:
        X_white = X_white[None, :]

    n_trials, n_channels, n_times = X_white.shape

    if estimate:
        ar_model.arma2psd(hold=True)
        ar_model.psd[-1] = 1. / ar_model.psd[-1]

        # ar_model.ordar *= 2

        if zero_phase:
            ar_model.psd[-1] = np.sqrt(ar_model.psd[-1])
        ar_model.estimate()

    # else:
    #     from numpy.polynomial.polynomial import polyfromroots
    #     # XXX not correct
    #     AR = np.hstack((np.ones(1), ar_model.AR_))
    #     poles = np.roots(AR)
    #
    #     n_poles = len(poles)
    #     reduced_polynomial = np.zeros((n_poles, n_poles), dtype='complex')
    #     for i in range(n_poles):
    #         idx_without_i = np.r_[np.arange(i), np.arange(i + 1, n_poles)]
    #         reduced_polynomial[i] = polyfromroots(poles[idx_without_i])
    #
    #     # ones = np.dot(reduced_polynomial.T, coefs)
    #     coefs = np.linalg.solve(reduced_polynomial.T, np.ones(n_poles))
    #
    #     ordar = 100
    #     poles_powered = np.power(poles[:, None], np.arange(ordar)[None, :])
    #     ar_model.AR_ = np.real(np.dot(coefs[None, :], poles_powered))[0]

    # apply the whitening twice (forward and backward) for zero-phase filtering
    if zero_phase:
        X_unwhite = [[
            ar_model.inverse(ar_model.inverse(X_np)[::-1])[::-1]
            for X_np in X_n
        ] for X_n in X_white]
    else:
        X_unwhite = [[ar_model.inverse(X_np) for X_np in X_n]
                     for X_n in X_white]

    X_unwhite = np.array(X_unwhite)
    assert X_unwhite.shape == X_white.shape

    if plot:
        # plot the Power Spectral Density (PSD) before/after
        ar_model.periodogram(
            X_white.reshape(-1, n_times), hold=False, mean_psd=True)
        ar_model.arma2psd(hold=True)
        ar_model.periodogram(
            X_unwhite.reshape(-1, n_times), hold=True, mean_psd=True)
        ar_model.plot('periodogram before/after unwhitening',
                      labels=['signal white', 'model AR',
                              'signal unwhite'], fscale='lin')
        plt.legend(loc='lower left')
    return X_unwhite


if __name__ == '__main__':
    sfreq = 300.
    n_times = 1000

    X_init = almost_pink_noise(n_times, slope=2.)
    X_init /= X_init.std()
    X_init = X_init[None, None, :]

    ar_model, X_white = whitening(X_init, sfreq, plot=True)
    X_unwhite = unwhitening(ar_model, X_white, plot=True)
    plt.show()

    X_unwhite /= X_unwhite.std()

    for x in [
            X_init[0, 0],
            X_white[0, 0],
            X_unwhite[0, 0]
    ]:
        plt.plot(x)
    plt.legend(['init', 'white', 'unwhite'])
    plt.show()
