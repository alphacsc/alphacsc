"""
Benchmark different solver of the same CSC univariate or multivariate problem.

This script needs the following packages:
    pip install pandas pyfftw

- Use bench_methods_run.py to run the benchmark.
  The results are saved in alphacsc/figures.
- Use bench_methods_plot.py to plot the results.
  The figures are saved in alphacsc/figures.
"""

from __future__ import print_function
from functools import partial
from pathlib import Path
import time
import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from alphacsc.update_d import update_d_block
from alphacsc.learn_d_z import learn_d_z
from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.datasets.mne_data import load_data

START = time.time()
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(30, 38)

##############################
# Parameters of the simulation
##############################

verbose = 1

# base string for the save names.
base_name = 'run_0'
# n_jobs for the parallel running of single core methods
n_jobs = 1
# max_iter for z step
z_max_iter = 1000
# number of outer iterations
n_iter_multi = 20
# tol for z step
z_tol = 1e-3
eps = 1e-3
# number of random states
n_states = 1
# loop over parameters
n_times_atom_list = [32]
n_atoms_list = [2]
n_channel_list = [1, 5, 204]
reg_list = [10.]
ds_init = "chunk"


######################################
# Functions compared in the benchmark
######################################


def run_fista(X, reg, n_iter, random_state, label):
    assert X.ndim == 2
    pobj, times, d_hat, z_hat, reg = learn_d_z(
        X, n_atoms, n_times_atom, func_d=update_d_block, solver_z='fista',
        solver_z_kwargs=dict(max_iter=2), reg=reg, n_iter=n_iter,
        random_state=random_state, ds_init=ds_init, n_jobs=1, verbose=verbose)

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


def run_l_bfgs(X, reg, n_iter, random_state, label, factr_d=1e7,
               factr_z=1e14):
    assert X.ndim == 2
    pobj, times, d_hat, z_hat, reg = learn_d_z(
        X, n_atoms, n_times_atom,
        func_d=update_d_block, solver_z='l-bfgs', solver_z_kwargs=dict(
            factr=factr_z), reg=reg, n_iter=n_iter, solver_d_kwargs=dict(
                factr=factr_d), random_state=random_state, ds_init=ds_init,
        n_jobs=1, verbose=verbose)

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


def run_multivariate(X, solver_z, reg, n_iter, random_state, label, rank1,
                     njobs):

    solver_z_kwargs = dict(max_iter=z_max_iter, tol=z_tol)
    pobj, times, d_hat, z_hat, reg = learn_d_z_multi(
        X, n_atoms, n_times_atom, solver_d='auto', solver_z=solver_z,
        uv_constraint='auto', eps=eps, solver_z_kwargs=solver_z_kwargs,
        reg=reg, solver_d_kwargs=dict(max_iter=100), n_iter=n_iter,
        random_state=random_state, raise_on_increase=False, D_init=ds_init,
        n_jobs=njobs, verbose=verbose, rank1=rank1)

    # remove the ds init duration
    times[0] = 0

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


def run_multichannel_gcd(X, reg, n_iter, random_state, label):
    if X.ndim == 2:
        X = X[:, None, :]

    return run_multivariate(X, "lgcd", reg, n_iter, random_state, label,
                            True, n_jobs)


def run_multichannel_dicodile(X, reg, n_iter, random_state, label, njobs=30):
    if X.ndim == 2:
        X = X[:, None, :]

    return run_multivariate(X, "dicodile", reg, n_iter, random_state, label,
                            True, njobs)


def run_multichannel_gcd_fullrank(X, reg, n_iter, random_state, label):
    assert X.ndim == 3

    return run_multivariate(X, "lgcd", reg, n_iter, random_state, label, False,
                            n_jobs)


def run_multichannel_dicodile_fullrank(X, reg, n_iter, random_state, label,
                                       njobs=30):
    assert X.ndim == 3

    return run_multivariate(X, "dicodile", reg, n_iter, random_state, label,
                            False, njobs)


def colorify(message, color=BLUE):
    """Change color of the standard output"""
    return ("\033[1;%dm" % color) + message + "\033[0m"


#########################################
# List of functions used in the benchmark
#########################################

n_iter = 100
methods_univariate = [
    [run_fista, 'Jas et al (2017) FISTA', n_iter],
    [run_l_bfgs, 'Jas et al (2017) LBFGS', n_iter],
    [run_multichannel_gcd, 'gcd', n_iter],
]

methods_multivariate = [
    [run_multichannel_gcd_fullrank, 'gcd fullrank', n_iter_multi],
    [partial(run_multichannel_dicodile_fullrank, njobs=5),
     'dicodile fullrank 5', n_iter_multi],
    [partial(run_multichannel_dicodile_fullrank, njobs=10),
     'dicodile fullrank 10', n_iter_multi],
    [partial(run_multichannel_dicodile_fullrank, njobs=30),
     'dicodile fullrank 30', n_iter_multi],
    [run_multichannel_gcd, 'gcd', n_iter_multi],
    [partial(run_multichannel_dicodile, njobs=5),
     'dicodile 5', n_iter_multi],
    [partial(run_multichannel_dicodile, njobs=10),
     'dicodile 10', n_iter_multi],
    [partial(run_multichannel_dicodile, njobs=30),
     'dicodile 30', n_iter_multi],
]


###################################
# Calling function of the benchmark
###################################


def one_run(X, X_shape, random_state, method, n_atoms, n_times_atom, reg):
    assert X.shape == X_shape
    func, label, n_iter = method
    current_time = time.time() - START
    msg = ('%s - %s: started at T=%.0f sec' % (random_state, label,
                                               current_time))
    print(colorify(msg, BLUE))

    if len(X_shape) == 2:
        n_trials, n_times = X.shape
        n_channels = 1
    else:
        n_trials, n_channels, n_times = X.shape

    # run the selected algorithm with one iter to remove compilation overhead
    # if dicodile, the workers are started but not stopped or reused, so
    # doubles the requirement for workers
    if 'dicodile' not in label:
        _, _, _, _ = func(X, reg, 1, random_state, label)

    # run the selected algorithm
    pobj, times, d_hat, z_hat = func(X, reg, n_iter, random_state, label)

    # store z_hat in a sparse matrix to reduce size
    for z in z_hat:
        z[z < 1e-3] = 0
    z_hat = [sp.csr_matrix(z) for z in z_hat]

    duration = time.time() - START - current_time
    current_time = time.time() - START
    msg = ('%s - %s: done in %.0f sec at T=%.0f sec' %
           (random_state, label, duration, current_time))
    print(colorify(msg, GREEN))
    return (random_state, label, np.asarray(pobj), np.asarray(times),
            np.asarray(d_hat), np.asarray(z_hat), n_atoms, n_times_atom,
            n_trials, n_times, n_channels, reg)


#################################################
# Iteration over parameter settings and functions
#################################################


if __name__ == '__main__':

    out_iterator = itertools.product(n_times_atom_list, n_atoms_list,
                                     n_channel_list, reg_list)

    figures_dir = Path('figures')
    figures_dir.mkdir(exist_ok=True)

    for params in out_iterator:
        n_times_atom, n_atoms, n_channels, reg = params
        msg = 'n_times_atom, n_atoms, n_channels, reg = ' + str(params)
        print(colorify(msg, RED))
        print(colorify('-' * len(msg), RED))

        all_results = []

        X, info = load_data(
            dataset='somato', epoch=False, n_jobs=n_jobs, n_splits=1
        )

        if n_channels == 1:
            X = X[:, 0, :]  # take only one channel
        elif n_channels is not None:
            X = X[:, :n_channels, :]

        X_shape = X.shape

        if n_channels == 1:
            methods = methods_univariate
        else:
            methods = methods_multivariate

        iterator = itertools.product(methods, range(n_states))
        if n_jobs == 1:
            results = [
                one_run(X, X_shape, random_state, method, n_atoms,
                        n_times_atom, reg)
                for method, random_state in iterator
            ]
        else:
            # run the methods for different random_state
            delayed_one_run = delayed(one_run)
            results = Parallel(n_jobs=n_jobs)(delayed_one_run(
                X, X_shape, random_state, method, n_atoms, n_times_atom,
                reg) for method, random_state in iterator)

        all_results.extend(results)

        file_name = base_name + str(params) + '.pkl'
        save_path = figures_dir / file_name

        all_results_df = pd.DataFrame(
            all_results, columns='random_state label pobj times d_hat '
            'z_hat n_atoms n_times_atom n_trials n_times n_channels reg'.
            split(' '))
        all_results_df.to_pickle(save_path)

    print('-- End of the script --')
