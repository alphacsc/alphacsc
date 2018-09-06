import os.path as op
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

import mne

from joblib import Memory

from alphacsc.utils import get_uv
from alphacsc.datasets.somato import load_data
from alphacsc.utils.viz import COLORS

matplotlib.rc('font', size=14)
mem = Memory(cachedir='.', verbose=0)

separate_figures = True

atoms_idx = 1
evoked = mne.read_evokeds(
    'examples_alphacsc/atom_multi_somato-ave.fif')[atoms_idx]

data_path = mne.datasets.somato.data_path()
subjects_dir = op.join(data_path, 'subjects')


def compute_dipole():

    fname_ave = 'examples_alphacsc/atom_multi_somato-ave.fif'

    fname_bem = op.join(subjects_dir, 'somato', 'bem',
                        'somato-5120-bem-sol.fif')
    fname_trans = op.join(data_path, 'MEG', 'somato',
                          'sef_raw_sss-trans.fif')

    evoked = mne.read_evokeds(fname_ave, baseline=None)[atoms_idx]
    evoked.crop(tmin=evoked.times[42], tmax=evoked.times[42])
    evoked.pick_types(meg=True, eeg=False)

    epochs = load_data(epoch=True, return_epochs=True)
    cov = mne.compute_covariance(epochs)

    # Fit a dipole
    dip = mne.fit_dipole(evoked, cov, fname_bem, fname_trans)[0]
    return dip


# Plot interesting atoms
uv_hat = get_uv(evoked.data[None, ...])

n_channels = evoked.info['nchan']
n_times_atom = uv_hat.shape[-1] - n_channels

v_hat = uv_hat[0, n_channels:]

times = np.arange(n_times_atom) / evoked.info['sfreq']

plt.figure(figsize=(4, 3))
plt.plot(times, v_hat.T, color=COLORS[0], linewidth=1.5)
plt.grid('on', linestyle='-', alpha=0.3)
plt.xlabel('Time (s)')
plt.gcf().savefig('figures/atoms_somato_a.pdf')

plt.figure(figsize=(3, 3))
mne.viz.plot_topomap(uv_hat[0, :n_channels], evoked.info,
                     axes=plt.gca())
plt.savefig('figures/atoms_somato_b.pdf')

plt.figure(figsize=(3, 3))
psd = np.abs(np.fft.rfft(v_hat, n=2 * n_times_atom)) ** 2
plt.xlabel('Frequencies (Hz)')
frequencies = np.linspace(0, evoked.info['sfreq'] / 2.0, len(psd))
plt.plot(frequencies, 10 * np.log10(psd), color=COLORS[0], linewidth=1.5)
plt.grid('on', linestyle='-', alpha=0.3)
plt.xlim(0, 25)
plt.ylim(-30, 20)
fmax = frequencies[np.argmax(psd)]
plt.axvline(fmax, linestyle='--', color=COLORS[1], linewidth=1.5)
plt.axvline(2 * fmax, linestyle='--', color=COLORS[1], linewidth=1.5)
plt.savefig('figures/atoms_somato_c.pdf')

fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(111, projection='3d')
fname_trans = op.join(data_path, 'MEG', 'somato', 'sef_raw_sss-trans.fif')
dip = compute_dipole()
dip.plot_locations(fname_trans, 'somato', subjects_dir, ax=ax,
                   mode='orthoview')
best_idx = np.argmax(dip.gof)
best_time = dip.times[best_idx]
print('Dipole fit (Highest GOF=%0.1f%%)' % dip.gof[best_idx])

# Boom
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set(xlabel='', ylabel='', zlabel='')
plt.suptitle('')
plt.savefig('figures/atoms_somato_d.pdf', bbox_inches='tight')
