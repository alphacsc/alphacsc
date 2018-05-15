# Run plot_sample_data_eyeblink.py before this to generate
# the evoked file

from os import path as op
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import mne

from alphacsc.utils import get_uv
from alphacsc.utils.viz import COLORS

data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
fname_ave = 'examples_multicsc/atom_multi_sample-ave.fif'
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
fname_bem = op.join(subjects_dir, 'sample', 'bem', 'sample-5120-bem-sol.fif')
fname_trans = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_raw-trans.fif')
fname_surf_lh = op.join(subjects_dir, 'sample', 'surf', 'lh.white')

evoked = mne.read_evokeds(fname_ave, baseline=None)[0]
evoked.pick_types(meg=True, eeg=False)

# Fit a dipole
dip = mne.fit_dipole(evoked, fname_cov, fname_bem, fname_trans)[0]

# Plot the result in 3D brain with the MRI image.
fig = plt.figure(figsize=plt.figaspect(2.))
ax = fig.add_subplot(2, 1, 1, projection='3d')
dip.plot_locations(fname_trans, 'sample', subjects_dir, ax=ax,
                   mode='orthoview')
best_idx = np.argmax(dip.gof)
best_time = dip.times[best_idx]
ax.set_title('Dipole fit (Highest GOF=%0.1f%%)' % dip.gof[best_idx])

# add PSD
ax = fig.add_subplot(2, 1, 2)
v_hat = get_uv(evoked.data[None, ...])[0, evoked.info['nchan']:]
psd = np.abs(np.fft.rfft(v_hat)) ** 2
frequencies = np.linspace(0, evoked.info['sfreq'] / 2.0, len(psd))
ax.semilogy(frequencies, psd, color=COLORS[0], linewidth=1.5)
ax.set(xlabel='Frequencies (Hz)', ylabel='Power Spectral Density')
ax.grid('on')
ax.set_xlim(0, 30)

plt.suptitle('')
plt.tight_layout()
plt.savefig('figures/dipole_sample.png', bbox='tight')
