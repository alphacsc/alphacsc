##########################################################################
# This code is taken from the pactools package available at
# https://github.com/pactools/pactools
#

import numpy as np
from scipy import signal, linalg, fftpack, fft


class Arma(object):
    """Estimator of ARMA model:
        y(t) + a(1)y(t-1) + ... + a(ordar)y(t-ordar) =
        b(0)e(t) + b(1)e(t-1) + ... + b(ordma)e(t-ordma)

    Parameters
    ----------
    ordar : int
        Order of the autoregressive part

    ordma : int
        Order of the moving average part

    block_length : int
        Length of each signal block, on which we estimate the spectrum

    fft_length : int or None
        Length of FFT, should be greater or equal to block_length.
        If None, it is set to block_length

    step : int or None
        Step between successive blocks
        If None, it is set to half the block length (i.e. 0.5 overlap)

    wfunc : function
        Function used to compute the weighting window on each block.
        Examples: np.ones, np.hamming, np.bartlett, np.blackman, ...

    fs : float
        Sampling frequency

    donorm : boolean
        If True, the amplitude is normalized

    """
    def __init__(self, ordar=2, ordma=0, block_length=1024, fft_length=None,
                 step=None, wfunc=np.hamming, fs=1., donorm=True):
        self.ordar = ordar
        self.ordma = ordma
        self.block_length = block_length
        self.fft_length = fft_length
        self.step = step
        self.wfunc = wfunc
        self.fs = fs
        self.donorm = donorm
        self.psd = []

    def check_params(self):
        # block_length
        if self.block_length <= 0:
            raise ValueError('Block length is negative: %s' %
                             (self.block_length, ))
        self.block_length = int(self.block_length)

        # fft_length
        if self.fft_length is None:
            fft_length = next_power2(self.block_length)
        else:
            fft_length = int(self.fft_length)
        if not is_power2(fft_length):
            raise ValueError('FFT length should be a power of 2')
        if fft_length < self.block_length:
            raise ValueError('Block length is greater than FFT length')

        # step
        if self.step is None:
            step = max(int(self.block_length // 2), 1)
        else:
            step = int(self.step)
        if step <= 0 or step > self.block_length:
            raise ValueError('Invalid step between blocks: %s' % (step, ))

        return fft_length, step

    def periodogram(self, signals, hold=False, mean_psd=False):
        """
        Computes the estimation (in dB) for each epoch in a signal

        Parameters
        ----------
        signals : array, shape (n_epochs, n_points)
            Signals from which one computes the power spectrum

        hold : boolean, default = False
            If True, the estimation is appended to the list of previous
            estimations, else, the list is emptied and only the current
            estimation is stored.

        mean_psd : boolean, default = False
            If True, the PSD is the mean PSD over all epochs.

        Returns
        -------
        psd : array, shape (n_epochs, n_freq) or (1, n_freq) if mean_psd
            Power spectrum estimated with a Welsh method on each epoch
            n_freq = fft_length // 2 + 1
        """
        fft_length, step = self.check_params()

        signals = np.atleast_2d(signals)
        n_epochs, n_points = signals.shape

        block_length = min(self.block_length, n_points)

        window = self.wfunc(block_length)
        n_epochs, tmax = signals.shape
        n_freq = fft_length // 2 + 1

        psd = np.zeros((n_epochs, n_freq))

        for i, sig in enumerate(signals):
            block = np.arange(block_length)

            # iterate on blocks
            count = 0
            while block[-1] < sig.size:
                psd[i] += np.abs(
                    fft(window * sig[block], fft_length, 0))[:n_freq] ** 2
                count = count + 1
                block = block + step
            if count == 0:
                raise IndexError(
                    'spectrum: first block has %d samples but sig has %d '
                    'samples' % (block[-1] + 1, sig.size))

            # normalize
            if self.donorm:
                scale = 1.0 / (count * (np.sum(window) ** 2))
            else:
                scale = 1.0 / count
            psd[i] *= scale

        if mean_psd:
            psd = np.mean(psd, axis=0)[None, :]

        if not hold:
            self.psd = []
        self.psd.append(psd)
        return psd

    def estimate(self, nbcorr=np.nan, numpsd=-1):
        fft_length, _ = self.check_params()
        if np.isnan((nbcorr)):
            nbcorr = self.ordar

        # -------- estimate correlation from psd
        full_psd = self.psd[numpsd]
        full_psd = np.c_[full_psd, np.conjugate(full_psd[:, :0:-1])]
        correl = fftpack.ifft(full_psd[0], fft_length, 0).real

        # -------- estimate AR part
        col1 = correl[self.ordma:self.ordma + nbcorr]
        row1 = correl[np.abs(
            np.arange(self.ordma, self.ordma - self.ordar, -1))]
        R = linalg.toeplitz(col1, row1)
        r = -correl[self.ordma + 1:self.ordma + nbcorr + 1]
        AR = linalg.solve(R, r)
        self.AR_ = AR

        # -------- estimate correlation of MA part

        # -------- estimate MA part
        if self.ordma == 0:
            sigma2 = correl[0] + np.dot(AR, correl[1:self.ordar + 1])
            self.MA = np.ones(1) * np.sqrt(sigma2)
        else:
            raise NotImplementedError(
                'arma: estimation of the MA part not yet implemented')

    def arma2psd(self, hold=False):
        """Compute the power spectral density of the ARMA model

        """
        fft_length, _ = self.check_params()
        arpart = np.concatenate((np.ones(1), self.AR_))
        psdar = np.abs(fftpack.fft(arpart, fft_length, 0)) ** 2
        psdma = np.abs(fftpack.fft(self.MA, fft_length, 0)) ** 2
        psd = psdma / psdar
        if not hold:
            self.psd = []
        self.psd.append(psd[None, :fft_length // 2 + 1])

    def inverse(self, sigin):
        """Apply the inverse ARMA filter to a signal

        sigin : input signal (ndarray)

        returns the filtered signal(ndarray)

        """
        arpart = np.concatenate((np.ones(1), self.AR_))
        return signal.fftconvolve(sigin, arpart, 'same')


def ai2ki(ar):
    """Convert AR coefficients to partial correlations
    (inverse Levinson recurrence)

    ar : AR models stored by columns

    returns the partial correlations (one model by column)

    """
    parcor = np.copy(ar)
    ordar, n_epochs, n_points = ar.shape
    for i in range(ordar - 1, -1, -1):
        if i > 0:
            parcor[0:i, :, :] -= (parcor[i:i + 1, :, :] *
                                  np.flipud(parcor[0:i, :, :]))
            parcor[0:i, :, :] *= 1.0 / (1.0 - parcor[i:i + 1, :, :] ** 2)
    return parcor


def ki2ai(parcor):
    """Convert parcor coefficients to autoregressive ones
    (Levinson recurrence)

    parcor : partial correlations stored by columns

    returns the AR models by columns

    """
    ar = np.zeros_like(parcor)
    ordar, n_epochs, n_points = parcor.shape
    for i in range(ordar):
        if i > 0:
            ar[0:i, :, :] += parcor[i:i + 1, :, :] * np.flipud(ar[0:i, :, :])
        ar[i, :, :] = parcor[i, :, :]

    # ok, at least in stationary models
    return ar


def is_power2(num):
    """Test if num is a power of 2. (int -> bool)"""
    num = int(num)
    return num != 0 and ((num & (num - 1)) == 0)


def next_power2(num):
    """Compute the smallest power of 2 >= to num.(float -> int)"""
    return 2 ** int(np.ceil(np.log2(num)))
