"""
==============================
CSC to learn LFP spiking atoms
==============================

Here, we show how CSC can be used to learn spiking
atoms from Local Field Potential (LFP) data [1].

[1] Hitziger, Sebastian, et al.
    Adaptive Waveform Learning: A Framework for Modeling Variability in
    Neurophysiological Signals. IEEE Transactions on Signal Processing (2017).
"""

###############################################################################
# First, let us fetch the data (~14 MB)
import os
from mne.utils import _fetch_file # noqa

url = ('https://github.com/hitziger/AWL/raw/master/Experiments/data/'
       'LFP_data_contiguous_1250_Hz.mat')
fname = './LFP_data_contiguous_1250_Hz.mat'
if not os.path.exists(fname):
    _fetch_file(url, fname)

###############################################################################
# It is a mat file, so we use scipy to load it
from scipy import io # noqa

data = io.loadmat(fname)
X, sfreq = data['X'].T, float(data['sfreq'])

###############################################################################
# And now let us look at the data
import numpy as np # noqa
import matplotlib.pyplot as plt # noqa

start, stop = 11000, 15000
times = np.arange(start, stop) / sfreq
plt.plot(times, X[0, start:stop], color='b')
plt.xlabel('Time (s)')
plt.ylabel(r'$\mu$ V')
plt.xlim([9., 12.])

###############################################################################
# and filter it using a convenient function from MNE. This will remove low
# frequency drifts, but we keep the high frequencies
from mne.filter import filter_data # noqa
X = filter_data(X.astype(np.float64), sfreq, l_freq=1, h_freq=None,
                fir_design='firwin')

###############################################################################
# Now, we define the parameters of our model.

reg = 6.0
n_times = 2500
n_times_atom = 350
n_trials = 100
n_atoms = 3
n_iter = 60
###############################################################################
# Let's stick to one random state for now, but if you want to learn how to
# select the random state, consult :ref:`this example
# <sphx_glr_auto_examples_plot_simulate_randomstate.py>`.
random_state = 10

###############################################################################
# Now, we epoch the trials

overlap = 0
starts = np.arange(0, X.shape[1] - n_times, n_times - overlap)
stops = np.arange(n_times, X.shape[1], n_times - overlap)

X_new = []
for idx, (start, stop) in enumerate(zip(starts, stops)):
    if idx >= n_trials:
        break
    X_new.append(X[0, start:stop])
X_new = np.vstack(X_new)
del X

###############################################################################
# We remove the mean and scale to unit variance.
X_new -= np.mean(X_new)
X_new /= np.std(X_new)

###############################################################################
# The convolutions can result in edge artifacts at the edges of the trials.
# Therefore, we discount the contributions from the edges by windowing the
# trials.
from numpy import hamming # noqa
X_new *= hamming(n_times)[None, :]

###############################################################################
# Of course, in a data-limited setting we want to use as much of the data as
# possible. If this is the case, you can set `overlap` to non-zero (for example
# half the epoch length).
#
# Now, we run regular CSC since the trials are not too noisy
from alphacsc import learn_d_z # noqa
pobj, times, d_hat, Z_hat = learn_d_z(X_new, n_atoms, n_times_atom, reg=reg,
                                      n_iter=n_iter, random_state=random_state,
                                      n_jobs=1)

###############################################################################
# Let's look at the atoms now.
plt.figure()
plt.plot(d_hat.T)
plt.show()
