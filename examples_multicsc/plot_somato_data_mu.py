import os
import time
import itertools
from functools import partial

import matplotlib
matplotlib.use('agg')

import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals.joblib import Memory
from scipy.signal import tukey

from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.utils.viz import plot_activations_density, COLORS

figure_path = 'figures'
mem = Memory(cachedir='.', verbose=0)

sfreq = 150.
n_times_atom = int(round(sfreq * 1.0))  # 1000. ms
reg_list = [29.]

# n_atoms = 30
n_atoms = 5
n_iter = 200
random_state = 0
n_jobs = 6

verbose = 2

event_id, tmin, tmax = 1, -2., 5.

# debug
if True:
    reg_list = [5]
    n_atoms = 2
    n_iter = 1
    n_states = 1
    n_jobs = 1


@mem.cache()
def load_data(sfreq=sfreq):
    # load the raw data
    data_path = os.path.join(mne.datasets.somato.data_path(), 'MEG', 'somato')
    raw = mne.io.read_raw_fif(
        os.path.join(data_path, 'sef_raw_sss.fif'), preload=True)

    # filter the signals
    raw.filter(2, 90., n_jobs=n_jobs)
    raw.notch_filter(np.arange(50, 101, 50), n_jobs=n_jobs)
    raw.resample(sfreq, npad='auto')

    # find the events
    events = mne.find_events(raw, stim_channel='STI 014')
    events[:, 0] -= raw.first_samp

    # pick channels and resample to sfreq sampling frequency
    raw.pick_types(
        meg='grad',
        eog=False,
        stim=False,
        eeg=False, )

    # get a numpy array from the raw object
    X = raw.get_data()

    # split into n_splits, to take advantage of multiprocessing
    n_channels, n_times = X.shape
    n_splits = 10
    n_times = n_times // n_splits
    X = np.reshape(X[:, :n_splits * n_times], (n_channels, n_splits, n_times))
    X = np.swapaxes(X, 0, 1)

    # normalization
    X *= tukey(n_times, alpha=0.02)
    X /= np.std(X)
    return X, raw.info, events


def time_string():
    """Get the time in a string """
    t = time.localtime(time.time())
    return '%4d-%02d-%02d-%02dh%02d' % (t[0], t[1], t[2], t[3], t[4])


def _run(random_state, reg, **kwargs):

    params = dict(
        X=X,
        n_atoms=n_atoms,
        n_times_atom=n_times_atom,
        reg=reg,
        eps=1e-5,
        uv_constraint='separate',
        solver_d='alternate_adaptive',
        solver_z_kwargs={'factr': 1e12},
        solver_d_kwargs={'max_iter': 300},
        loss='l2',
        verbose=verbose,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs)

    _, _, uv_init, _ = learn_d_z_multi(n_iter=0, **params)
    pobj, times, uv_hat, Z_hat = learn_d_z_multi(n_iter=n_iter, **params)

    return pobj, times, uv_hat, Z_hat, uv_init


def one_run(method, random_state, reg):
    name, func = method
    print('------------- running %s ----------------' % name)
    return func(random_state, reg)


X, info, events = load_data()
n_trials, n_channels, n_times = X.shape

method = ('chunk', partial(_run, D_init='chunk'))

all_methods = [method]

save_name = os.path.join(figure_path, 'somato')
save_name += '_' + method[0]
save_name += '_' + time_string()

results = [one_run(method, random_state, reg_list[0])]

labels = [
    '%s_r%s_%d' % (method[0], reg_list[0], random_state)
]

if True:
    # ------------ plot many representations of the atoms init/final

    def plot_psd_final(res, ax, i_atom):
        pobj, times, uv_hat, Z_hat, uv_init = res
        v_hat = uv_hat[i_atom, n_channels:]
        psd = np.abs(np.fft.rfft(v_hat)) ** 2
        frequencies = np.linspace(0, sfreq / 2.0, len(psd))
        ax.plot(frequencies, psd, color=COLORS[i_atom % len(COLORS)])
        ax.set(xlabel='Frequencies (Hz)', title='PSD atom final')
        ax.grid('on')
        ax.set_xlim(0, 30)

    def plot_atom_final(res, ax, i_atom):
        pobj, times, uv_hat, Z_hat, uv_init = res
        v_hat = uv_hat[i_atom, n_channels:]
        ax.plot(
            np.arange(v_hat.size) / sfreq, v_hat,
            color=COLORS[i_atom % len(COLORS)])
        ax.set(xlabel='Time (sec)', title='Atom final')
        ax.grid('on')

    def plot_topo_final(res, ax, i_atom):
        pobj, times, uv_hat, Z_hat, uv_init = res
        u_hat = uv_hat[i_atom, :n_channels]
        mne.viz.plot_topomap(u_hat, info, axes=ax, show=False)
        ax.set(title='Topomap final')

    all_plots = [
        plot_psd_final,
        plot_atom_final,
        plot_topo_final
    ]

    n_plots = len(all_plots)

    for label, res in zip(labels, results):

        figsize = (2 + n_atoms * 3, n_plots * 3)
        fig, axes = plt.subplots(n_plots, n_atoms, figsize=figsize,
                                 squeeze=False)
        for i_atom in range(n_atoms):
            for i_plot, plot_func in enumerate(all_plots):
                ax = axes[i_plot, i_atom]
                plot_func(res, ax, i_atom)
        plt.tight_layout()
        fig.savefig(save_name + '_' + label + '_multiplot.png')

plt.close('all')
# plt.show()
