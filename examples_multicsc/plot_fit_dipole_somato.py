# Run plot_somato_data_mu.py before this script

import os
from os import path as op
import numpy as np
import matplotlib.pyplot as plt

import mne

from alphacsc.datasets.somato import load_data
from alphacsc.utils import get_uv
from alphacsc.utils.viz import COLORS

data_path = mne.datasets.somato.data_path()
subjects_dir = op.join(data_path, 'subjects')

fname_ave = 'examples_multicsc/atom_multi_somato-ave.fif'

fname_raw = os.path.join(data_path, 'sef_raw_sss.fif')
fname_bem = op.join(subjects_dir, 'somato', 'bem', 'somato-5120-bem-sol.fif')
fname_trans = op.join(data_path, 'MEG', 'somato',
                      'sef_raw_sss-trans.fif')
fname_surf_lh = op.join(subjects_dir, 'somato', 'surf', 'lh.white')

atom_idx = 4
evoked = mne.read_evokeds(fname_ave, baseline=None)[atom_idx]
evoked.pick_types(meg=True, eeg=False)

epochs = load_data(epoch=True, return_epochs=True)
cov = mne.compute_covariance(epochs)

# Fit a dipole
dip = mne.fit_dipole(evoked, cov, fname_bem, fname_trans)[0]

# Plot the result in 3D brain with the MRI image.
fig = plt.figure(figsize=plt.figaspect(2.))
ax = fig.add_subplot(2, 1, 1, projection='3d')
dip.plot_locations(fname_trans, 'somato', subjects_dir, ax=ax,
                   mode='orthoview')
best_idx = np.argmax(dip.gof)
best_time = dip.times[best_idx]
ax.set_title('Dipole fit (Highest GOF=%0.1f%%)' % dip.gof[best_idx])

# add PSD
ax = fig.add_subplot(2, 1, 2)
v_hat = get_uv(evoked.data[None, ...])[0, evoked.info['nchan']:]
psd = np.abs(np.fft.rfft(v_hat)) ** 2
frequencies = np.linspace(0, evoked.info['sfreq'] / 2.0, len(psd))
ax.plot(frequencies, 10 * np.log10(psd), color=COLORS[0], linewidth=1.5)
ax.set(xlabel='Frequencies (Hz)', ylabel='Power Spectral Density (dB)')
ax.grid('on')
ax.set_xlim(0, 40)
ax.axvline(8.6, linestyle='--', color=COLORS[1])
ax.axvline(17.2, linestyle='--', color=COLORS[1])

plt.suptitle('')
plt.tight_layout()
plt.savefig('figures/dipole_somato.png', bbox='tight')
