import os
import itertools
import numpy as np
from scipy.signal import tukey
import matplotlib.pyplot as plt
from sklearn.externals.joblib import Memory, Parallel, delayed

import os.path as op
import mne

from alphacsc.learn_d_z_multi import learn_uv_z
from alphacsc.viz import plot_activations_density


mem = Memory(cachedir='.', verbose=0)

n_atoms = 25
sfreq = 300
n_times_atom = int(sfreq * .2)


@mem.cache()
def load_data(sfreq=sfreq):
    data_path = os.path.join(mne.datasets.somato.data_path(), 'MEG', 'somato')
    raw = mne.io.read_raw_fif(
        os.path.join(data_path, 'sef_raw_sss.fif'), preload=True)
    raw.filter(15., 90., n_jobs=n_jobs)
    raw.notch_filter(np.arange(50, 101, 50), n_jobs=n_jobs)

    events = mne.find_events(raw, stim_channel='STI 014')

    event_id, tmin, tmax = 1, -1., 3.
    baseline = (None, 0)
    picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                           stim=False)

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        picks=picks, baseline=baseline, reject=dict(
                            grad=4000e-13, eog=350e-6), preload=True)
    epochs.pick_types(meg='grad', eog=False)
    epochs.resample(sfreq, npad='auto')

    # define n_chan, n_trials, n_times
    X = epochs.get_data()
    n_trials, n_chan, n_times = X.shape
    X *= tukey(n_times, alpha=0.1)[None, None, :]
    X /= np.std(X)
    return X


X = load_data()
n_trials, n_chan, n_times = X.shape
ncols = int(np.ceil(n_atoms / 5.))
nrows = min(5, n_atoms)

plt.close('all')
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, num='atoms',
                         figsize=(10, 8), sharex=True, sharey=True)
fig_Z, axes_Z = plt.subplots(nrows=nrows, ncols=ncols, num='Z',
                             figsize=(10, 8), sharex=True, sharey=True)
fig_Xk, axes_Xk = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
fig.tight_layout()
fig_Z.tight_layout()
# fig_topo.canvas.draw()
# fig_topo, axes_topo = plt.subplots(1, n_atoms, figsize=(12, 3))
# if n_atoms == 1:
#     axes, axes_topo, axes_Z = [axes_topo], [axes], [axes_Z]


def callback(X, uv_hat, Z_hat, reg):
    # for idx in range(n_atoms):
    #     mne.viz.plot_topomap(uv_hat[idx, :n_chan], epochs.info,
    #                          axes=axes_topo[idx], show=False)
    #     axes_topo[idx].set_title('atom %d' % idx)
    if axes[0, 0].lines == []:
        for k, ax in enumerate(axes.ravel()):
            ax.plot(np.arange(n_times_atom) / sfreq, uv_hat[k, n_chan:].T)
            ax.grid(True)
    else:
        for ax, uv in zip(axes.ravel(), uv_hat):
            ax.lines[0].set_ydata(uv[n_chan:])
            ax.relim()  # make sure all the data fits
            ax.autoscale_view(True, True, True)
    plot_activations_density(Z_hat, n_times_atom, sfreq=sfreq,
                             plot_activations=False, axes=axes_Z,
                             bandwidth='auto')

    n_channels = X.shape[1]
    X_hat = [np.convolve(Zk, uvk[n_channels:])
             for Zk, uvk in zip(Z_hat[:, 0], uv_hat)]
    if axes_Xk[0, 0].lines == []:
        for k, ax in enumerate(axes_Xk.flatten()):
            ax.plot(np.arange(X_hat[k].shape[-1]) / sfreq,
                    X_hat[k])
            ax.grid(True)
    else:
        for ax, x in zip(axes_Xk.flatten(), X_hat):
            ax.lines[0].set_ydata(x)
            ax.relim()  # make sure all the data fits
            ax.autoscale_view(True, True, True)
    fig.canvas.draw()
    # fig_topo.canvas.draw()
    fig_Xk.canvas.draw()
    fig_Z.canvas.draw()
    plt.pause(.001)


def run_one(func, *args, **kwargs):
    return func(*args, **kwargs)

_run_one_cached = mem.cache(run_one)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Programme to launch experiemnt')
    parser.add_argument('--multi', action='store_true',
                        help='Launch a grid search on reg/gcd_max_iter')
    parser.add_argument('--njobs', type=int, default=4,
                        help='Number of CPU one for multiprocessing exp.')
    args = parser.parse_args()

    if args.multi:
        gcd_iter = [10, 100, 1000]
        reg = [.5, 1, 5, 10, 20]
        grid = itertools.product(reg, gcd_iter)
        with Parallel(n_jobs=args.njobs) as parallel:
            delayed_run_one = delayed(_run_one_cached)
            res = parallel(delayed_run_one(
                learn_uv_z, X, n_atoms, n_times_atom, random_state=42,
                n_iter=50, n_jobs=1, reg=REG, eps=1e-8, solver_z="gcd",
                solver_z_kwargs={'max_iter': GCD},
                solver_d='alternate_adaptive',
                solver_d_kwargs={'max_iter': 100},
                uv_init='kmeans', uv_constraint='separate',
                algorithm='batch')
                           for REG, GCD in grid)

    else:
        pobj, times, uv_hat, Z_hat = learn_uv_z(
            X, n_atoms, n_times_atom, random_state=42, n_iter=50,
            n_jobs=args.njobs, reg=20, eps=1e-8,
            solver_z="gcd", solver_z_kwargs={'max_iter': 10000},
            solver_d='alternate_adaptive', solver_d_kwargs={'max_iter': 100},
            uv_init='kmeans', uv_constraint='separate',
            callback=callback, algorithm='batch')

    import IPython
    IPython.embed()