import mne
import matplotlib.pyplot as plt
import numpy as np

from multicsc.utils.viz import COLORS

data = np.load('examples_multicsc/multi_sample-ave.npz')
info = mne.io.read_info('examples_multicsc/info_sample.fif')
n_channels = data['n_channels']
uv_hat = data['uv_hat']
sfreq = data['sfreq']
Z_hat = data['Z_hat']

n_times_atom = uv_hat.shape[-1] - n_channels

atoms_idx = [7, 9, 18, 4]
# Plot interesting atoms
times = np.arange(n_times_atom) / sfreq
fig = plt.figure(figsize=(12, 7))
for idx, atom_idx in enumerate(atoms_idx):

    ax1 = plt.subplot2grid((len(atoms_idx), 9), (idx, 0), colspan=3)
    ax1.plot(times, uv_hat[atoms_idx[idx], n_channels:],
             color=COLORS[idx % len(COLORS)])
    ax1.grid('on', linestyle='-', alpha=0.3)
    ax1.set_yticklabels([])

    ax2 = plt.subplot2grid((len(atoms_idx), 9), (idx, 3), colspan=4)
    ax2.grid('on', linestyle='-', alpha=0.3)
    times_Z = np.arange(2231, 6693) / sfreq
    ax2.plot(times_Z, Z_hat[atom_idx, 0, 2231:6693],
             color=COLORS[idx % len(COLORS)])
    ax2.set_ylim(0, 0.2)

    ax3 = plt.subplot2grid((len(atoms_idx), 9), (idx, 7), colspan=2)
    mne.viz.plot_topomap(uv_hat[atom_idx, :n_channels], info,
                         axes=ax3)

    if idx == 0:
        ax1.set_title('A. Temporal waveform', fontsize=16)
        ax2.set_title('B. Activations', fontsize=16)
        ax3.set_title('C. Spatial pattern', fontsize=16)

ax1.set_xlabel('Time (s)')
ax2.set_xlabel('Time (s)')

fig.tight_layout()
# fig.subplots_adjust(hspace=0.2)
fig.savefig('figures/atoms_sample.pdf')
