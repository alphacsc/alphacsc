import matplotlib.pyplot as plt
import numpy as np
import mne

from alphacsc.utils import get_uv
from alphacsc.utils.viz import COLORS

atoms_idx = 4
evoked = mne.read_evokeds('examples_multicsc/atom_multi_somato-ave.fif')[atoms_idx]

# Plot interesting atoms
uv_hat = get_uv(evoked.data[None, ...])

n_channels = evoked.info['nchan']
n_times_atom = uv_hat.shape[-1] - n_channels

times = np.arange(n_times_atom) / evoked.info['sfreq']
fig = plt.figure(figsize=(7, 3))
atoms_idx = [4]
for idx, _ in enumerate(atoms_idx):

    ax1 = plt.subplot2grid((len(atoms_idx), 3), (idx, 0), colspan=2)
    ax1.plot(times, uv_hat[idx, n_channels:].T, color=COLORS[0])
    ax1.grid('on')

    ax2 = plt.subplot2grid((len(atoms_idx), 3), (idx, 2), colspan=1)
    mne.viz.plot_topomap(uv_hat[idx, :n_channels], evoked.info,
                         axes=ax2)

    ax1.set_title('A. Temporal waveform', fontsize=16)
    ax2.set_title('B. Spatial pattern', fontsize=16)
ax1.set_xlabel('Time (s)')
fig.tight_layout()
fig.savefig('figures/atoms_somato.pdf')
