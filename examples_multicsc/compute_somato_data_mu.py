import os
import time
from functools import partial

import matplotlib
matplotlib.use('agg')

import mne
from mne.io import write_info

import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals.joblib import Memory

from multicsc.utils.viz import COLORS
from multicsc.datasets.somato import load_data
from multicsc.learn_d_z_multi import learn_d_z_multi

figure_path = 'figures'
mem = Memory(cachedir='.', verbose=0)

sfreq = 150.
n_times_atom = int(round(sfreq * 1.0))  # 1000. ms
reg_list = [29.]

n_atoms = 25
n_iter = 60
random_state = 0
n_jobs = 10

verbose = 10

# debug
if False:
    reg_list = [5]
    n_atoms = 2
    n_iter = 1
    n_states = 1
    n_jobs = 1


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
        solver_z="lgcd",
        uv_constraint='separate',
        solver_d='alternate_adaptive',
        solver_z_kwargs={'factr': 1e12, 'max_iter': 1000},
        solver_d_kwargs={'max_iter': 300},
        loss='l2',
        verbose=verbose,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs)

    _, _, uv_init, _ = learn_d_z_multi(n_iter=0, **params)
    pobj, times, uv_hat, z_hat = learn_d_z_multi(n_iter=n_iter, **params)

    return pobj, times, uv_hat, z_hat, uv_init


def one_run(method, random_state, reg):
    name, func = method
    print('------------- running %s ----------------' % name)
    return func(random_state, reg)


X, info = load_data(epoch=True, sfreq=sfreq)
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


def plot_psd_final(res, ax, i_atom):
    pobj, times, uv_hat, z_hat, uv_init = res
    v_hat = uv_hat[i_atom, n_channels:]
    psd = np.abs(np.fft.rfft(v_hat)) ** 2
    frequencies = np.linspace(0, sfreq / 2.0, len(psd))
    ax.plot(frequencies, psd, color=COLORS[i_atom % len(COLORS)])
    ax.set(xlabel='Frequencies (Hz)', title='Power Spectral Density')
    ax.grid('on')
    ax.set_xlim(0, 30)


def plot_atom_final(res, ax, i_atom):
    pobj, times, uv_hat, z_hat, uv_init = res
    v_hat = uv_hat[i_atom, n_channels:]
    ax.plot(
        np.arange(v_hat.size) / sfreq, v_hat,
        color=COLORS[i_atom % len(COLORS)])
    ax.set(xlabel='Time (sec)', title='Learned temporal waveform')
    ax.grid('on')


def plot_topo_final(res, ax, i_atom):
    pobj, times, uv_hat, z_hat, uv_init = res
    u_hat = uv_hat[i_atom, :n_channels]
    mne.viz.plot_topomap(u_hat, info, axes=ax, show=False)
    ax.set(title='Learned spatial pattern')


all_plots = [
    plot_psd_final,
    plot_atom_final,
    plot_topo_final
]

n_plots = len(all_plots)

for label, res in zip(labels, results):

    figsize = (n_plots * 3.5, 2 + n_atoms * 3)
    fig, axes = plt.subplots(n_atoms, n_plots, figsize=figsize,
                             squeeze=False)
    for i_atom in range(n_atoms):
        for i_plot, plot_func in enumerate(all_plots):
            ax = axes[i_atom, i_plot]
            plot_func(res, ax, i_atom)
    plt.tight_layout()
    fig.savefig(save_name + '_' + label + '_multiplot.pdf')

plt.close('all')
plt.show()

pobj, times, uv_hat, z_hat, uv_init = res

np.savez('examples_multicsc/multi_somato-ave.npz', z_hat=z_hat,
         uv_hat=uv_hat, sfreq=info['sfreq'], n_channels=n_channels)
write_info('examples_multicsc/info_somato.fif', info)
