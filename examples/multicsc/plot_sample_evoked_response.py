# -*- coding: utf-8 -*-
"""
=====================================================================
Extracting artifact and evoked response atoms from the sample dataset
=====================================================================

This example illustrates how to learn rank-1 [1]_ atoms on the multivariate
sample dataset from :code:`mne`. We display a selection of atoms, featuring
heartbeat and eyeblink artifacts, three atoms of evoked responses, and a
non-sinusoidal oscillation.

.. [1] Dupré La Tour, T., Moreau, T., Jas, M., & Gramfort, A. (2018).
    `Multivariate Convolutional Sparse Coding for Electromagnetic Brain Signals
    <https://arxiv.org/abs/1805.09654v2>`_. Advances in Neural Information
    Processing Systems (NIPS).
"""

# Authors: Thomas Moreau <thomas.moreau@inria.fr>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

###############################################################################
# Let us first define the parameters of our model.

# sample frequency
sfreq = 150.

# Define the shape of the dictionary
n_atoms = 40
n_times_atom = int(round(sfreq * 1.0))  # 1000. ms

# Regularization parameter which control sparsity
reg = 0.1

# number of processors for parallel computing
n_jobs = 5

###############################################################################
# Next, we define the parameters for multivariate CSC

from alphacsc import GreedyCDL
cdl = GreedyCDL(
    # Shape of the dictionary
    n_atoms=n_atoms,
    n_times_atom=n_times_atom,
    # Request a rank1 dictionary with unit norm temporal and spatial maps
    rank1=True,
    uv_constraint='separate',
    # apply a temporal window reparametrization
    window=True,
    # at the end, refit the activations with fixed support and no reg to unbias
    unbiased_z_hat=True,
    # Initialize the dictionary with random chunk from the data
    D_init='chunk',
    # rescale the regularization parameter to be a percentage of lambda_max
    lmbd_max="scaled",
    reg=reg,
    # Number of iteration for the alternate minimization and cvg threshold
    n_iter=100,
    eps=1e-4,
    # solver for the z-step
    solver_z="lgcd",
    solver_z_kwargs={'tol': 1e-2,
                     'max_iter': 10000},
    # solver for the d-step
    solver_d='alternate_adaptive',
    solver_d_kwargs={'max_iter': 300},
    # sort atoms by explained variances
    sort_atoms=True,
    # Technical parameters
    verbose=1,
    random_state=0,
    n_jobs=n_jobs)


###############################################################################
# Here, we load the MNE sample data. The data is not epoched yet,
# but we split it into 12 parts to make the most of multiple processors during
# the model fitting.

import os.path as op

import mne
from mne.datasets import sample

data_path = mne.datasets.sample.data_path()
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(raw_fname, preload=True)

###############################################################################
# The only temporal filter we apply on our data is a notch filter.
import numpy as np
raw.notch_filter(np.arange(60, 181, 60), n_jobs=n_jobs)
raw.pick_types(meg='grad', eeg=False, eog=False, stim=False)

###############################################################################
# We extract the multivariate time series as a numpy array

n_splits = 12
X = raw.get_data()
n_channels, n_times = X.shape
n_times = n_times // n_splits
X_split = X[:, :n_splits * n_times]
X_split = X_split.reshape(n_channels, n_splits, n_times).swapaxes(0, 1)

n_splits, n_channels, n_times = X_split.shape

###############################################################################
# Then we apply a Tukey window to the time series to avoid edge artifacts
# when learning the atoms
from scipy.signal import tukey

X_split *= tukey(n_times, alpha=0.1)[None, None, :]
X_split /= np.std(X_split)

###############################################################################
# Fit the model and learn rank1 atoms
cdl.fit(X_split)

###############################################################################
# Then we call the `transform` method, which returns the sparse codes
# associated with X, without changing the dictionary learned during the `fit`.
# Note that we transform on the *unsplit* data so that the sparse codes
# reflect the original data and not the windowed data.
z_hat = cdl.transform(X[None, :])

###############################################################################
# Display a selection of atoms. We recognize a heartbeat artifact, an
# eyeblink artifact, three atoms of evoked responses, and a non-sinusoidal
# oscillation.

import mne
import numpy as np
import matplotlib.pyplot as plt

from alphacsc.utils.convolution import construct_X_multi
from alphacsc.viz.epoch import plot_evoked_surrogates

# preselected atoms of interest
plotted_atoms = [1, 0, 3, 16, 20, 7]

n_plots = 3  # number of plots by atom
n_columns = min(6, len(plotted_atoms))
split = int(np.ceil(len(plotted_atoms) / n_columns))
figsize = (4 * n_columns, 3 * n_plots * split)
fig, axes = plt.subplots(n_plots * split, n_columns, figsize=figsize)
for ii, kk in enumerate(plotted_atoms):

    # Select the axes to display the current atom
    print("\rDisplaying {}-th atom".format(kk), end='', flush=True)
    i_row, i_col = ii // n_columns, ii % n_columns
    it_axes = iter(axes[i_row * n_plots:(i_row + 1) * n_plots, i_col])

    # Select the current atom
    u_k = cdl.u_hat_[kk]
    v_k = cdl.v_hat_[kk]

    # Plot the spatial map of the atom using mne topomap
    ax = next(it_axes)
    mne.viz.plot_topomap(u_k, info, axes=ax, show=False)
    ax.set(title="Spatial pattern %d" % (kk, ))

    # Plot the temporal pattern of the atom
    ax = next(it_axes)
    t = np.arange(n_times_atom) / sfreq
    ax.plot(t, v_k)
    ax.set_xlim(0, n_times_atom / sfreq)
    ax.set(xlabel='Time (sec)', title="Temporal pattern %d" % kk)

    # Plot the power spectral density (PSD)
    ax = next(it_axes)
    psd = np.abs(np.fft.rfft(v_k, n=256)) ** 2
    frequencies = np.linspace(0, sfreq / 2.0, len(psd))
    ax.semilogy(frequencies, psd, label='PSD', color='k')
    ax.set(xlabel='Frequencies (Hz)', title="Power spectral density %d" % kk)
    ax.grid(True)
    ax.set_xlim(0, 30)
    ax.set_ylim(1e-4, 1e2)
    ax.legend()

fig.tight_layout()

###############################################################################
# Display the evoked reconstructed envelope:
# For each atom (columns), and for each event (rows), we compute the envelope
# of the reconstructed signal, align it with respect to the event onsets, and
# take the average. For some atoms, the activations are correlated with the
# events, leading to a large evoked envelope. The gray area corresponds to
# not statistically significant values.

from copy import deepcopy
from alphacsc.utils.signal import fast_hilbert

# time window around the events
t_lim = (-0.1, 0.5)

n_plots = len(np.atleast_1d(info['event_id']))
n_columns = min(6, len(plotted_atoms))
split = int(np.ceil(len(plotted_atoms) / n_columns))
figsize = (4 * n_columns, 3 * n_plots * split)
fig, axes = plt.subplots(n_plots * split, n_columns, figsize=figsize)

for ii, kk in enumerate(plotted_atoms):

    # Select the axes to display the current atom
    print("\rDisplaying {}-th atom".format(kk), end='', flush=True)
    i_row, i_col = ii // n_columns, ii % n_columns
    it_axes = iter(axes[i_row * n_plots:(i_row + 1) * n_plots, i_col])

    # Select the current atom
    v_k = cdl.v_hat_[kk]
    v_k_1 = np.r_[[1], v_k][None]
    z_k = z_hat[:, kk:kk + 1]
    X_k = construct_X_multi(z_k, v_k_1, n_channels=1)[0, 0]

    # compute the 'envelope' of the reconstructed signal X_k
    correlation = np.abs(fast_hilbert(X_k))

    # loop over all events IDs
    for this_event_id in event_id:
        this_events = events[[events][:, 2] == this_event_id]
        # plotting function
        ax = next(it_axes)
        plot_evoked_surrogates(correlation, info=this_info, t_lim=t_lim, ax=ax,
                               n_jobs=n_jobs, label='event %d' % this_event_id)
        ax.set(xlabel='Time (sec)', title="Evoked envelope %d" % kk)

fig.tight_layout()
