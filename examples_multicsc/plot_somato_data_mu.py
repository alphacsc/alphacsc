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

atoms_idx = 4
evoked = mne.read_evokeds(
    'examples_multicsc/atom_multi_somato-ave.fif')[atoms_idx]

data_path = mne.datasets.somato.data_path()
subjects_dir = op.join(data_path, 'subjects')


@mem.cache()
def compute_dipole():

    fname_ave = 'examples_multicsc/atom_multi_somato-ave.fif'

    fname_bem = op.join(subjects_dir, 'somato', 'bem',
                        'somato-5120-bem-sol.fif')
    fname_trans = op.join(data_path, 'MEG', 'somato',
                          'sef_raw_sss-trans.fif')

    atom_idx = 4
    evoked = mne.read_evokeds(fname_ave, baseline=None)[atom_idx]
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
fig = plt.figure(figsize=(12, 3))
atoms_idx = [4]

ax1 = plt.subplot2grid((10, 4), (0, 0), colspan=1, rowspan=9)
ax1.plot(times, -v_hat.T, color=COLORS[0], linewidth=1.5)
ax1.grid('on', linestyle='-', alpha=0.3)

ax2 = plt.subplot2grid((10, 4), (0, 1), colspan=1, rowspan=9)
mne.viz.plot_topomap(uv_hat[0, :n_channels], evoked.info,
                     axes=ax2)

ax3 = plt.subplot2grid((10, 4), (0, 2), colspan=1, rowspan=9)
psd = np.abs(np.fft.rfft(v_hat)) ** 2
ax3.set(xlabel='Frequencies (Hz)')
frequencies = np.linspace(0, evoked.info['sfreq'] / 2.0, len(psd))
ax3.plot(frequencies, 10 * np.log10(psd), color=COLORS[0], linewidth=1.5)
ax3.grid('on', linestyle='-', alpha=0.3)
ax3.set_xlim(0, 25)
ax3.set_ylim(-30, 20)
fmax = frequencies[np.argmax(psd)]
ax3.axvline(fmax, linestyle='--', color=COLORS[1], linewidth=1.5)
ax3.axvline(2 * fmax, linestyle='--', color=COLORS[1], linewidth=1.5)

ax4 = plt.subplot2grid((10, 4), (0, 3), colspan=1, projection='3d', rowspan=10)
fname_trans = op.join(data_path, 'MEG', 'somato', 'sef_raw_sss-trans.fif')
dip = compute_dipole()
dip.plot_locations(fname_trans, 'somato', subjects_dir, ax=ax4,
                   mode='orthoview')
best_idx = np.argmax(dip.gof)
best_time = dip.times[best_idx]
print('Dipole fit (Highest GOF=%0.1f%%)' % dip.gof[best_idx])

# Boom
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_zticks([])
ax4.set(xlabel='', ylabel='', zlabel='')

ax1.set_xlabel('Time (s)')
plt.suptitle('')
plt.tight_layout(w_pad=0.)
plt.tight_layout(w_pad=0.)

if separate_figures:
    labels = ['a', 'b', 'c', 'd']
    for ax, label in zip([ax1, ax2, ax3, ax4], labels):
        extent = ax.get_tightbbox(
            fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig('figures/atoms_somato_%s.pdf' % label, bbox_inches=extent,
                    dpi=10)
else:
    fig.savefig('figures/atoms_somato.pdf')
