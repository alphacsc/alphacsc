import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import tukey

import os.path as op
import mne

from alphacsc.learn_d_z_multi import learn_d_z_multi

n_jobs = 2
n_atoms = 5

data_path = op.join(mne.datasets.somato.data_path(), 'MEG', 'somato')
raw = mne.io.read_raw_fif(op.join(data_path, 'sef_raw_sss.fif'),
                          preload=True)
raw.filter(10., 30., n_jobs=n_jobs)
events = mne.find_events(raw, stim_channel='STI 014')

event_id, tmin, tmax = 1, -1., 3.
baseline = (None, 0)
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True, stim=False)

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=baseline, reject=dict(grad=4000e-13, eog=350e-6),
                    preload=True)
epochs.pick_types(meg='grad', eog=False)
epochs.resample(150., npad='auto')

# define n_chan, n_trials, n_times
X = epochs.get_data()
n_trials, n_chan, n_times = X.shape
n_times_atom = int(round(epochs.info['sfreq'] * 0.2))  # 200. ms

# make windows
X *= tukey(n_times, alpha=0.1)[None, None, :]
X /= np.std(X)

plt.close('all')
fig, axes = plt.subplots(nrows=n_atoms, num='atoms', figsize=(10, 8),
                         sharex=True)
fig_Z, axes_Z = plt.subplots(nrows=n_atoms, num='Z', figsize=(10, 8),
                             sharex=True, sharey=True)
fig_topo, axes_topo = plt.subplots(1, n_atoms, figsize=(12, 3))
if n_atoms == 1:
    axes, axes_topo, axes_Z = [axes_topo], [axes], [axes_Z]


def callback(X, uv_hat, Z_hat, reg):
    for idx in range(n_atoms):
        mne.viz.plot_topomap(uv_hat[idx, :n_chan], epochs.info,
                             axes=axes_topo[idx], show=False)
        axes_topo[idx].set_title('atom %d' % idx)
    if axes[0].lines == []:
        for k in range(n_atoms):
            axes[k].plot(np.arange(n_times_atom) / epochs.info['sfreq'],
                         uv_hat[k, n_chan:].T)
            axes[k].grid(True)
    else:
        for ax, uv in zip(axes, uv_hat):
            ax.lines[0].set_ydata(uv[n_chan:])
            ax.relim()  # make sure all the data fits
            ax.autoscale_view(True, True, True)
    if axes_Z[0].lines == []:
        for k in range(n_atoms):
            axes_Z[k].plot(np.arange(Z_hat.shape[-1]) / epochs.info['sfreq'],
                           Z_hat[k, 0])
            axes_Z[k].grid(True)
    else:
        for ax, z in zip(axes_Z, Z_hat[:, 0]):
            ax.lines[0].set_ydata(z)
            ax.relim()  # make sure all the data fits
            ax.autoscale_view(True, True, True)
    fig.canvas.draw()
    fig_topo.canvas.draw()
    fig_Z.canvas.draw()
    plt.pause(.001)


pobj, times, uv_hat, Z_hat = learn_d_z_multi(
    X, n_atoms, n_times_atom, random_state=42, n_iter=100, n_jobs=1, reg=20.0,
    eps=1e-3, solver_z_kwargs={'factr': 1e12},
    solver_d_kwargs={'max_iter': 300}, uv_constraint='separate',
    solver_d='alternate', callback=callback)
