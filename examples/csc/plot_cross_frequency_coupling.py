"""
==================================================================
Extracting cross-frequency coupling waveforms from rodent LFP data
==================================================================

This example illustrates how to learn univariate atoms on a univariate
time-serie. The data is a single LFP channel recorded on a rodent's striatum
[1]_. Interestingly in this time-serie, the high frequency oscillations around
80 Hz are modulated in amplitude by the low-frequency oscillation around 3 Hz,
a phenomenon known as cross-frequency coupling (CFC).

The convolutional sparse coding (CSC) model is able to learn the prototypical
waveforms of the signal, on which we can clearly see the CFC.

.. [1] G. Dallérac, M. Graupner, J. Knippenberg, R. C. R. Martinez,
    T. F. Tavares, L. Tallot, N. El Massioui, A. Verschueren, S. Höhn,
    J.B. Bertolus, et al. Updating temporal expectancy of an aversive event
    engages striatal plasticity under amygdala control.
    Nature Communications, 8:13920, 2017
"""

# Authors: Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

###############################################################################
# Let us first load the data sample.

import mne
import numpy as np
import matplotlib.pyplot as plt

# sample frequency
sfreq = 350.

# We load the signal. It is an LFP channel recorded on a rodent's striatum.
data = np.load('../rodent_striatum.npy')
print(data.shape)

###############################################################################
# As the data contains severe artifacts between t=0 and t=100, we use a
# section not affected by artifacts.

data = data[:, 35000:]

# We also remove the slow drift, which accounts for a lot of variance.
data = mne.filter.filter_data(data, sfreq, 1, None)

# To make the most of parallel computing, we split the data into trials.
data = data.reshape(50, -1)
data /= data.std()

###############################################################################
# This sample contains CFC between 3 Hz and 80 Hz. This phenomenon can be
# described with a comodulogram, computed for instance with the `pactools
# <http://pactools.github.io/>`_ Python library.

from pactools import Comodulogram

comod = Comodulogram(fs=sfreq, low_fq_range=np.arange(0.2, 10.2, 0.2),
                     low_fq_width=2., method='duprelatour')
comod.fit(data)
comod.plot()
plt.show()

###############################################################################
# We fit a CSC model on the data.

from alphacsc import learn_d_z

params = dict(
    n_atoms=3,
    n_times_atom=int(sfreq * 1.0),  # 1000. ms
    reg=5.,
    n_iter=10,
    solver_z='l-bfgs',
    solver_z_kwargs=dict(factr=1e9),
    solver_d_kwargs=dict(factr=1e2),
    random_state=42,
    n_jobs=5,
    verbose=1)

_, _, d_hat, z_hat, _ = learn_d_z(data, **params)

###############################################################################
# Plot the temporal patterns. Interestingly, we obtain prototypical
# waveforms of the signal on which we can clearly see the CFC.

n_atoms, n_times_atom = d_hat.shape
n_columns = min(6, n_atoms)
n_rows = int(np.ceil(n_atoms // n_columns))
figsize = (4 * n_columns, 3 * n_rows)
fig, axes = plt.subplots(n_rows, n_columns, figsize=figsize, sharey=True)
axes = axes.ravel()

for kk in range(n_atoms):
    ax = axes[kk]
    time = np.arange(n_times_atom) / sfreq
    ax.plot(time, d_hat[kk], color='C%d' % kk)
    ax.set_xlim(0, n_times_atom / sfreq)
    ax.set(xlabel='Time (sec)', title="Temporal pattern %d" % kk)
    ax.grid(True)

fig.tight_layout()
plt.show()
