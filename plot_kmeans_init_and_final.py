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

from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.viz import plot_activations_density

figure_path = 'figures'
mem = Memory(cachedir='.', verbose=0)
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

sfreq = 150.
n_times_atom = int(round(sfreq * 1.0))  # 1000. ms
reg_list = np.arange(5, 33, 3)

n_atoms = 4
n_iter = 100
n_states = 1
n_jobs = 10

verbose = 1


@mem.cache()
def load_data(sfreq=sfreq):
    data_path = os.path.join(mne.datasets.somato.data_path(), 'MEG', 'somato')
    raw = mne.io.read_raw_fif(
        os.path.join(data_path, 'sef_raw_sss.fif'), preload=True)
    raw.filter(2., 90., n_jobs=n_jobs)
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
    return X, epochs.info


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
        eps=1e-4,
        uv_constraint='separate',
        solver_d='alternate_adaptive',
        solver_z_kwargs={'factr': 1e12},
        solver_d_kwargs={'max_iter': 300},
        loss='stdw',
        loss_params=dict(gamma=0.005, sakoe_chiba_band=10),
        verbose=verbose,
        random_state=random_state,
        n_jobs=1,
        **kwargs, )

    _, _, uv_init, _ = learn_d_z_multi(n_iter=0, **params)
    pobj, times, uv_hat, Z_hat = learn_d_z_multi(n_iter=n_iter, **params)

    return pobj, times, uv_hat, Z_hat, uv_init


def one_run(method, random_state, reg):
    name, func = method
    print('------------- running %s ----------------' % name)
    return func(random_state, reg)


X, epochs_info = load_data()
n_trials, n_channels, n_times = X.shape

all_methods_ = [
    ('chunk', partial(_run, uv_init='chunk')),
    ('kmedoid_0', partial(_run, uv_init='kmeans', kmeans_params=dict(
        max_iter=0, distances='roll_inv'))),
    ('kmedoid_100', partial(_run, uv_init='kmeans', kmeans_params=dict(
        max_iter=100, distances='roll_inv'))),
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

    if True:
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
                    color=colors[i_atom % len(colors)])
                ax.set_xlabel('Time (s)')
                ax.set_title(label)
                ax.grid('on')
        plt.tight_layout()
        fig.savefig(save_name + '_atoms_init.png')

    if True:
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
                    color=colors[i_atom % len(colors)])
                ax.set_xlabel('Time (s)')
                ax.set_title(label)
                ax.grid('on')
        plt.tight_layout()
        fig.savefig(save_name + '_atoms_end.png')

    if False:
        # ------------ plot the atoms of the method number i_method
        fig, axes = plt.subplots(n_atoms, 1, sharex=True, sharey=True,
                                 figsize=(10, 2 + n_atoms * 2))
        axes = np.atleast_1d(axes).reshape(n_atoms)
        i_method = 0
        label = labels[i_method]
        res = results[i_method]
        pobj, times, uv_hat, Z_hat, uv_init = res
        v_hat = uv_hat[:, n_channels:]
        for i_atom in range(n_atoms):
            ax = axes[i_atom]
            ax.plot(
                np.arange(v_hat.shape[1]) / sfreq, v_hat[i_atom],
                color=colors[i_atom % len(colors)])
            ax.set_xlabel('Time (s)')
            ax.grid('on')
        plt.tight_layout()
        fig.savefig(save_name + '_atoms_end_%d.png' % i_method)

    if True:
        # ------------ compute the best pobj over all methods
        best_pobj = np.inf
        for label, res in zip(labels, results):
            pobj, times, uv_hat, Z_hat, uv_init = res
            pobj_min = np.array(pobj).min()
            if pobj_min < best_pobj:
                best_pobj = pobj_min

        # ------------ plot the convergence curve
        fig = plt.figure()
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        # method_colors = [
        #     c for c in colors for _ in range(n_states) for _ in reg_list
        # ]
        method_colors = [c for c in colors for _ in range(n_states)]
        for label, res, color in zip(labels, results, method_colors):
            pobj, times, uv_hat, Z_hat, uv_init = res
            plt.semilogx(
                np.cumsum(times)[::2], pobj[::2], '.-', alpha=0.5, label=label,
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
        # ------------ plot the activations of one method
        i_method = 0
        pobj, times, uv_hat, Z_hat, uv_init = results[i_method]
        n_trials_plot = min(24, Z_hat.shape[1])
        n_trials, n_channels, n_times = X.shape
        fig, axs = plt.subplots(-(-n_trials_plot // 3), 3, sharex=True,
                                sharey=True, figsize=(16, 12))
        axs = axs.ravel()
        for i in range(n_trials_plot):
            for j in range(Z_hat.shape[0]):
                activations = Z_hat[j, i, :]
                time_instants = np.arange(activations.size) / sfreq
                selection = activations > 0.1

                axs[i].plot(time_instants[selection], activations[selection],
                            '.', color=colors[j])

        plt.tight_layout()
        fig.savefig(save_name + '_activations.png')

    if False:
        # ------------ plot the activations of all methods
        n_trials_plot = min(12, Z_hat.shape[1])
        fig, axes = plt.subplots(
            len(results), n_atoms, sharex=True, figsize=(2 + n_atoms * 3,
                                                         len(results) * 3))
        axes = np.atleast_1d(axes).reshape(len(results), n_atoms)
        for i_func, (label, res) in enumerate(zip(labels, results)):
            pobj, times, uv_hat, Z_hat, uv_init = res
            for i_atom in range(n_atoms):
                ax = axes[i_func, i_atom]
                for i_trial in range(n_trials_plot):
                    activations = Z_hat[i_atom, i_trial]
                    time_instants = np.arange(activations.size) / sfreq
                    selection = activations > 0.1
                    ax.plot(time_instants[selection], activations[selection],
                            '.', color=colors[i_atom % len(colors)])
                ax.set_title(label)
        plt.tight_layout()
        fig.savefig(save_name + '_activations_all.png')

    if True:
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

    if True:
        fig, axes = plt.subplots(
            len(results), n_atoms, figsize=(2 + n_atoms * 3, len(results) * 3))
        axes = np.atleast_1d(axes).reshape(len(results), n_atoms)

        for i_func, (label, res) in enumerate(zip(labels, results)):
            pobj, times, uv_hat, Z_hat, uv_init = res
            for i_atom in range(n_atoms):
                ax = axes[i_func, i_atom]
                uv_hat[i_atom, :n_channels]
                mne.viz.plot_topomap(uv_hat[i_atom, :n_channels], epochs_info,
                                     axes=ax, show=False)
        plt.tight_layout()
        fig.savefig(save_name + '_topo_all.png')

    plt.close('all')
    # plt.show()
