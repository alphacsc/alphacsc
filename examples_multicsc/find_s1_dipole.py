import os.path as op
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

import mne

from joblib import Memory

from multicsc.utils import get_uv
from multicsc.datasets.somato import load_data
from multicsc.utils.viz import COLORS

matplotlib.rc('font', size=14)
mem = Memory(cachedir='.', verbose=0)

separate_figures = True

atoms_idx = 1
evoked = mne.read_evokeds(
    'examples_multicsc/atom_multi_somato-ave.fif')[atoms_idx]

data_path = mne.datasets.somato.data_path()
subjects_dir = op.join(data_path, 'subjects')


@mem.cache()
def compute_dipole(atoms_idx):

    fname_ave = 'examples_multicsc/atom_multi_somato-ave.fif'

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

fig = plt.figure(figsize=(8, 8))

for atoms_idx in range(25):
    ax = fig.add_subplot(5, 5, atoms_idx + 1, projection='3d')
    fname_trans = op.join(data_path, 'MEG', 'somato', 'sef_raw_sss-trans.fif')
    dip = compute_dipole(atoms_idx)
    dip.plot_locations(fname_trans, 'somato', subjects_dir, ax=ax,
                       mode='orthoview')
    best_idx = np.argmax(dip.gof)
    best_time = dip.times[best_idx]
    print('Dipole fit (GOF=%0.1f%%)' % dip.gof[best_idx])

    # Boom
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set(xlabel='', ylabel='', zlabel='', title='Atom %s' % atoms_idx)
plt.suptitle('')
