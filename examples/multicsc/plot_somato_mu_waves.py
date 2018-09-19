"""
===========================================================
Extracting :math:`\mu`-wave from the somato-sensory dataset
===========================================================

This example illustrates how to learn rank1 atoms on the somato-sensorymotor
dataset from :code:`mne`. The displayed results highlight the presence of
:math:`\mu`-waves located in the SI cortex.

"""

# Authors: Thomas Moreau <thomas.moreau@inria.fr>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

###############################################################################
# Let us first define the parameters of our model.

sfreq = 150.

# Define the shape of the dictionary
n_atoms = 25
n_times_atom = int(round(sfreq * 1.0))  # 1000. ms

###############################################################################
# Next, we define the parameters for multivariate CSC

from alphacsc import BatchCDL  # noqa
cdl = BatchCDL(
    # Shape of the dictionary
    n_atoms=n_atoms,
    n_times_atom=n_times_atom,
    # Request a rank1 dictionary with unit norm temporal and spatial maps
    rank1=True, uv_constraint='separate',
    # Initialize the dictionary with random chunk from the data
    D_init='chunk',
    # rescale the regularization parameter to be 20% of lambda_max
    lmbd_max="scaled", reg=.2,
    # Number of iteration for the alternate minimization and cvg threshold
    n_iter=100, eps=1e-4,
    # solver for the z-step
    solver_z="lgcd", solver_z_kwargs={'tol': 1e-2, 'max_iter': 1000},
    # solver for the d-step
    solver_d='alternate_adaptive', solver_d_kwargs={'max_iter': 300},
    # Technical parameters
    verbose=1, random_state=0, n_jobs=6)


###############################################################################
# Here, we load the data from the somato-sensory dataset and preprocess them

from alphacsc.datasets.somato import load_data  # noqa
X, info = load_data(epoch=True, sfreq=sfreq)


###############################################################################
# Fit the model and learn rank1 atoms
cdl.fit(X)

###############################################################################
# Display the 4-th atom, which displays a :math:`\mu`-waveform in its temporal
# pattern.

import mne  # noqa
import numpy as np  # noqa
import matplotlib.pyplot as plt  # noqa

i_atom = 4
n_plots = 3
figsize = (n_plots * 3.5, 5)
fig, axes = plt.subplots(1, n_plots, figsize=figsize, squeeze=False)

# Plot the spatial map of the learn atom using mne topomap
ax = axes[0, 0]
u_hat = cdl.u_hat_[i_atom]
mne.viz.plot_topomap(u_hat, info, axes=ax, show=False)
ax.set(title='Learned spatial pattern')

# Plot the temporal pattern of the learn atom
ax = axes[0, 1]
v_hat = cdl.v_hat_[i_atom]
t = np.arange(v_hat.size) / sfreq
ax.plot(t, v_hat)
ax.set(xlabel='Time (sec)', title='Learned temporal waveform')
ax.grid(True)

# Plot the psd of the time atom
ax = axes[0, 2]
psd = np.abs(np.fft.rfft(v_hat)) ** 2
frequencies = np.linspace(0, sfreq / 2.0, len(psd))
ax.semilogy(frequencies, psd)
ax.set(xlabel='Frequencies (Hz)', title='Power Spectral Density')
ax.grid(True)
ax.set_xlim(0, 30)

plt.tight_layout()
plt.show()
