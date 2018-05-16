import mne
import matplotlib.pyplot as plt
import numpy as np

from alphacsc.utils.viz import COLORS

data = np.load('examples_multicsc/multi_sample-ave.npz')
info = mne.io.read_info('examples_multicsc/info_sample.fif')
n_channels = data['n_channels']
uv_hat = data['uv_hat']
sfreq = data['sfreq']
Z_hat = data['Z_hat']

n_times_atom = uv_hat.shape[-1] - n_channels

atoms_idx = [7, 18, 4]
# Plot interesting atoms
times = np.arange(n_times_atom) / sfreq
fig = plt.figure(figsize=(7, 8))
for idx, atom_idx in enumerate(atoms_idx):

    ax1 = plt.subplot2grid((len(atoms_idx) + 1, 3), (idx, 0), colspan=2)
    ax1.plot(times, uv_hat[atoms_idx[idx], n_channels:].T, color=COLORS[0])
    ax1.grid('on')

    ax2 = plt.subplot2grid((len(atoms_idx) + 1, 3), (idx, 2), colspan=1)
    mne.viz.plot_topomap(uv_hat[atom_idx, :n_channels], info,
                         axes=ax2)

    if idx == 0:
        ax1.set_title('A. Temporal waveform', fontsize=16)
        ax2.set_title('B. Spatial pattern', fontsize=16)
ax1.set_xlabel('Time (s)')
ax3 = plt.subplot2grid((len(atoms_idx) + 1, 3), (idx + 1, 0), colspan=3)
ax3.grid('on')
times = np.arange(4462) / sfreq
ax3.plot(times, Z_hat[atoms_idx[2], 0, :4462], color=COLORS[0])
ax3.set_title('            C. Activations', fontsize=16)
ax3.set_xlabel('Time (s)')
# time_diff = np.diff(np.where(
#     Z_hat[atoms_idx[2], 0, :] > 0.15)[0]) / raw.info['sfreq']
# time_diff = time_diff[time_diff > 0.1]
# print('Average pulse %f / min' % (1 / time_diff.mean() * 60))

# fig.tight_layout()
fig.subplots_adjust(hspace=0.4)
fig.savefig('figures/atoms_sample.pdf')
