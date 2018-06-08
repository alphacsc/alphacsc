import os
import time
import itertools
from functools import partial

import matplotlib
matplotlib.use('agg')

import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals.joblib import Memory, Parallel, delayed
from scipy.signal import tukey

from multicsc.learn_d_z_multi import learn_d_z_multi
from multicsc.utils.viz import plot_activations_density, COLORS

figure_path = 'figures'
mem = Memory(cachedir='.', verbose=0)

sfreq = 150.
n_times_atom = int(round(sfreq * 1.0))  # 1000. ms
reg_list = np.arange(5, 35, 3)

n_atoms = 30
n_iter = 200
n_states = 1
n_jobs = 10

verbose = 2

event_id, tmin, tmax = 1, -2., 5.

# debug
if False:
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


def make_epochs(Z_hat, info, events, n_times_atom):
    """Make Epochs on the activations of atoms.
    n_atoms, n_splits, n_times_valid = Z_hat.shape
    n_atoms, n_trials, n_times_epoch = Z_hat_epoched.shape
    """
    n_atoms, n_splits, n_times_valid = Z_hat.shape
    n_times = n_times_valid + n_times_atom - 1
    # pad with zeros
    padding = np.zeros((n_atoms, n_splits, n_times_atom - 1))
    Z_hat = np.concatenate([Z_hat, padding], axis=2)
    # reshape into an unique time-serie per atom
    Z_hat = np.reshape(Z_hat, (n_atoms, n_splits * n_times))

    # create trials around the events, using mne
    new_info = mne.create_info(ch_names=n_atoms, sfreq=info['sfreq'])
    rawarray = mne.io.RawArray(data=Z_hat, info=new_info)
    epochs = mne.Epochs(rawarray, events, event_id, tmin, tmax)
    Z_hat_epoched = np.swapaxes(epochs.get_data(), axis1=0, axis2=1)
    return Z_hat_epoched


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
        n_jobs=1,
        **kwargs, )

    _, _, uv_init, _ = learn_d_z_multi(n_iter=0, **params)
    pobj, times, uv_hat, Z_hat = learn_d_z_multi(n_iter=n_iter, **params)

    Z_hat = make_epochs(Z_hat, info, events, n_times_atom)

    return pobj, times, uv_hat, Z_hat, uv_init


def one_run(method, random_state, reg):
    name, func = method
    print('------------- running %s ----------------' % name)
    return func(random_state, reg)


X, info, events = load_data()
n_trials, n_channels, n_times = X.shape

all_methods_ = [
    ('chunk', partial(_run, D_init='chunk')),
    ('ssa', partial(_run, D_init='ssa')),
    ('roll_inv_0', partial(_run, D_init='kmeans', kmeans_params=dict(
        max_iter=0, distances='roll_inv'))),
    ('trans_inv_0', partial(_run, D_init='kmeans', kmeans_params=dict(
        max_iter=0, distances='trans_inv'))),
]

for method in all_methods_:
    all_methods = [method]

    save_name = os.path.join(figure_path, 'somato')
    save_name += '_' + method[0]
    save_name += '_' + time_string()

    delayed_one_run = delayed(one_run)
    results = Parallel(n_jobs=n_jobs)(
        delayed_one_run(method, random_state, reg)
        for method, random_state, reg in itertools.product(
            all_methods, range(n_states), reg_list))

    labels = [
        '%s_r%s_%d' % (method[0], reg, random_state)
        for method, random_state, reg in itertools.product(
            all_methods, range(n_states), reg_list)
    ]

    if False:
        # ------------ plot the atoms init
        fig, axes = plt.subplots(
            len(results), n_atoms, sharex=True, sharey=True,
            figsize=(2 + n_atoms * 3, len(results) * 3))
        axes = np.atleast_1d(axes).reshape(len(results), n_atoms)
        for i_func, (label, res) in enumerate(zip(labels, results)):
            pobj, times, uv_hat, Z_hat, uv_init = res
            v_init = uv_init[:, n_channels:]
            for i_atom in range(n_atoms):
                ax = axes[i_func, i_atom]
                ax.plot(
                    np.arange(v_init.shape[1]) / sfreq, v_init[i_atom],
                    color=COLORS[i_atom % len(COLORS)])
                ax.set_xlabel('Time (s)')
                ax.set_title(label)
                ax.grid('on')
        plt.tight_layout()
        fig.savefig(save_name + '_atoms_init.png')

    if False:
        # ------------ plot the atoms
        fig, axes = plt.subplots(
            len(results), n_atoms, sharex=True, sharey=True,
            figsize=(2 + n_atoms * 3, len(results) * 3))
        axes = np.atleast_1d(axes).reshape(len(results), n_atoms)
        for i_func, (label, res) in enumerate(zip(labels, results)):
            pobj, times, uv_hat, Z_hat, uv_init = res
            v_hat = uv_hat[:, n_channels:]
            for i_atom in range(n_atoms):
                ax = axes[i_func, i_atom]
                ax.plot(
                    np.arange(v_hat.shape[1]) / sfreq, v_hat[i_atom],
                    color=COLORS[i_atom % len(COLORS)])
                ax.set_xlabel('Time (s)')
                ax.set_title(label)
                ax.grid('on')
        plt.tight_layout()
        fig.savefig(save_name + '_atoms_end.png')

    if True:
        # ------------ plot the convergence curve

        # compute the best pobj over all methods
        best_pobj = np.inf
        for label, res in zip(labels, results):
            pobj, times, uv_hat, Z_hat, uv_init = res
            pobj_min = np.array(pobj).min()
            if pobj_min < best_pobj:
                best_pobj = pobj_min

        fig = plt.figure()
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        # method_colors = [
        #     c for c in COLORS for _ in range(n_states) for _ in reg_list
        # ]
        method_colors = [
            col
            for col, _ in zip(results, itertools.cycler(COLORS))
            for _ in range(n_states)
        ]
        for label, res, color in zip(labels, results, method_colors):
            pobj, times, uv_hat, Z_hat, uv_init = res
            plt.semilogx(
                np.cumsum(times), pobj, '.-', alpha=0.5, label=label,
                color=color)
        plt.xlabel('Time (s)')
        plt.ylabel('Objective value')
        plt.legend()

        plt.gca().tick_params(axis='x', which='both', bottom='off', top='off')
        plt.gca().tick_params(axis='y', which='both', left='off', right='off')
        plt.tight_layout()
        plt.grid('on')
        fig.savefig(save_name + '_convergence.png')

    if False:
        # ------------ plot the activations of all methods
        fig, axes = plt.subplots(
            len(results), n_atoms, sharex=True, sharey=True,
            figsize=(2 + n_atoms * 3, len(results) * 3))
        axes = np.atleast_1d(axes).reshape(len(results), n_atoms)
        for i_func, (label, res) in enumerate(zip(labels, results)):
            pobj, times, uv_hat, Z_hat, uv_init = res
            plot_activations_density(Z_hat, n_times_atom, sfreq=sfreq,
                                     axes=axes[i_func], plot_activations=False)
            axes[i_func][0].set_title(label)
        plt.tight_layout()
        fig.savefig(save_name + '_activations_all.png')

    if False:
        # ------------ plot the topomap
        fig, axes = plt.subplots(
            len(results), n_atoms, figsize=(2 + n_atoms * 3, len(results) * 3))
        axes = np.atleast_1d(axes).reshape(len(results), n_atoms)

        for i_func, (label, res) in enumerate(zip(labels, results)):
            pobj, times, uv_hat, Z_hat, uv_init = res
            for i_atom in range(n_atoms):
                ax = axes[i_func, i_atom]
                mne.viz.plot_topomap(uv_hat[i_atom, :n_channels], info,
                                     axes=ax, show=False)
        plt.tight_layout()
        fig.savefig(save_name + '_topo_all.png')

    if True:
        # ------------ plot many representations of the atoms init/final

        def plot_psd_init(res, ax, i_atom):
            pobj, times, uv_hat, Z_hat, uv_init = res
            v_init = uv_init[i_atom, n_channels:]
            psd = np.abs(np.fft.rfft(v_init)) ** 2
            frequencies = np.linspace(0, sfreq / 2.0, len(psd))
            ax.plot(frequencies, psd, color=COLORS[i_atom % len(COLORS)])
            ax.set(xlabel='Frequencies (Hz)', title='PSD atom init')
            ax.grid('on')
            ax.set_xlim(0, 30)

        def plot_psd_final(res, ax, i_atom):
            pobj, times, uv_hat, Z_hat, uv_init = res
            v_hat = uv_hat[i_atom, n_channels:]
            psd = np.abs(np.fft.rfft(v_hat)) ** 2
            frequencies = np.linspace(0, sfreq / 2.0, len(psd))
            ax.plot(frequencies, psd, color=COLORS[i_atom % len(COLORS)])
            ax.set(xlabel='Frequencies (Hz)', title='PSD atom final')
            ax.grid('on')
            ax.set_xlim(0, 30)

        def plot_atom_init(res, ax, i_atom):
            pobj, times, uv_hat, Z_hat, uv_init = res
            v_init = uv_init[i_atom, n_channels:]
            ax.plot(
                np.arange(v_init.size) / sfreq, v_init,
                color=COLORS[i_atom % len(COLORS)])
            ax.set(xlabel='Time (sec)', title='Atom init')
            ax.grid('on')

        def plot_atom_final(res, ax, i_atom):
            pobj, times, uv_hat, Z_hat, uv_init = res
            v_hat = uv_hat[i_atom, n_channels:]
            ax.plot(
                np.arange(v_hat.size) / sfreq, v_hat,
                color=COLORS[i_atom % len(COLORS)])
            ax.set(xlabel='Time (sec)', title='Atom final')
            ax.grid('on')

        def plot_topo_init(res, ax, i_atom):
            pobj, times, uv_hat, Z_hat, uv_init = res
            u_init = uv_init[i_atom, :n_channels]
            mne.viz.plot_topomap(u_init, info, axes=ax, show=False)
            ax.set(title='Topomap init')

        def plot_topo_final(res, ax, i_atom):
            pobj, times, uv_hat, Z_hat, uv_init = res
            u_hat = uv_hat[i_atom, :n_channels]
            mne.viz.plot_topomap(u_hat, info, axes=ax, show=False)
            ax.set(title='Topomap final')

        def plot_activations(res, ax, i_atom):
            pobj, times, uv_hat, Z_hat, uv_init = res
            z_hat = Z_hat[i_atom][None, ...]
            plot_activations_density(z_hat, n_times_atom, sfreq=sfreq,
                                     axes=[ax],
                                     colors=[COLORS[i_atom % len(COLORS)]])
            ax.axvline(-tmin, c='k')
            ax.axvline(-tmin + 0.5, c='k', ls='--')
            ax.axvline(-tmin + 1.5, c='k', ls='--')
            ax.set(xlabel='Time (sec)', title='Activations final')

        all_plots = [
            plot_psd_init,
            plot_atom_init,
            plot_topo_init,
            plot_psd_final,
            plot_atom_final,
            plot_topo_final,
            plot_activations,
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
