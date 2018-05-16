# Run compute_sample_data.py before this to generate
# the evoked file

import matplotlib
from os import path as op
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import mne

from alphacsc.utils.viz import COLORS

matplotlib.rc('font', size=14)

data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
fname_info = 'examples_multicsc/info_sample.fif'
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
fname_bem = op.join(subjects_dir, 'sample', 'bem', 'sample-5120-bem-sol.fif')
fname_trans = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_raw-trans.fif')
fname_surf_lh = op.join(subjects_dir, 'sample', 'surf', 'lh.white')

atom_idx = 7

info = mne.io.read_info(fname_info)
data = np.load('examples_multicsc/multi_sample-ave.npz')
n_channels, uv_hat = data['n_channels'], data['uv_hat']

evoked = mne.EvokedArray(uv_hat[atom_idx, :n_channels][:, None], info)

# Fit a dipole
dip = mne.fit_dipole(evoked, fname_cov, fname_bem, fname_trans)[0]

# Plot the result in 3D brain with the MRI image.
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 2, 1, projection='3d')
dip.plot_locations(fname_trans, 'sample', subjects_dir, ax=ax,
                   mode='orthoview')
best_idx = np.argmax(dip.gof)
best_time = dip.times[best_idx]
ax.set_yticks([])
ax.set_xticks([])
ax.set_zticks([])
ax.set(xlabel='', ylabel='', zlabel='')
plt.title('Dipole fit', y=1.08)
print('GOF = %f' % dip.gof[best_idx])

# add PSD
ax = fig.add_subplot(1, 2, 2)
v_hat = uv_hat[atom_idx, n_channels:]
psd = np.abs(np.fft.rfft(v_hat)) ** 2
frequencies = np.linspace(0, evoked.info['sfreq'] / 2.0, len(psd))
ax.plot(frequencies, 10 * np.log10(psd),
        color=COLORS[0], linewidth=1.5)
ax.set(xlabel='Frequencies (Hz)', ylabel='Power Spectral Density (dB)')
ax.grid('on')
ax.set_xlim(0, 40)

plt.suptitle('')
plt.tight_layout()
plt.savefig('figures/dipole_sample.png', bbox='tight')
