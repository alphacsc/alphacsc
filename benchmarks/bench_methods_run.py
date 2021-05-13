"""
Benchmark different solver of the same CSC univariate or multivariate problem.

This script needs the following packages:
    pip install pandas pyfftw
    pip install alphacsc/other/sporco

- Use bench_methods_run.py to run the benchmark.
  The results are saved in alphacsc/figures.
- Use bench_methods_plot.py to plot the results.
  The figures are saved in alphacsc/figures.
"""

from __future__ import print_function
import os
import time
import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

import alphacsc.other.heide_csc as CSC
from sporco.admm.cbpdndl import ConvBPDNDictLearn

from alphacsc.update_d import update_d_block
from alphacsc.learn_d_z import learn_d_z
from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.datasets.mne_data import load_data
from alphacsc.init_dict import init_dictionary
from alphacsc.utils.dictionary import get_uv

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
# number of random states
n_states = 1
# loop over parameters
n_times_atom_list = [32]
n_atoms_list = [2]
n_channel_list = [1]
reg_list = [10.]


######################################
# Functions compared in the benchmark
######################################


def run_admm(X, ds_init, reg, n_iter, random_state, label, max_it_d=10,
             max_it_z=10):
    # admm with the following differences
    # - positivity constraints
    # - different init
    # - d step and z step are swapped
    tol = np.float64(1e-3)
    size_kernel = ds_init.shape
    assert size_kernel[1] % 2 == 1
    [d, z, Dz, list_obj_val, times_admm] = CSC.learn_conv_sparse_coder(
        X, size_kernel, max_it=n_iter, tol=tol, random_state=random_state,
        lambda_prior=reg, ds_init=ds_init, verbose=verbose, max_it_d=max_it_d,
        max_it_z=max_it_z)

    # z.shape = (n_trials, n_atoms, n_times + 2 * n_times_atom)
    z = z[:, :, 2 * n_times_atom:-2 * n_times_atom]
    z = z.swapaxes(0, 1)
    # z.shape = (n_atoms, n_trials, n_times - 2 * n_times_atom)

    return list_obj_val, np.cumsum(times_admm)[::2], d, z


def run_cbpdn(X, ds_init, reg, n_iter, random_state, label):
    # Use only one thread in fft for fair comparison
    import sporco.linalg
    sporco.linalg.pyfftw_threads = 1

    if X.ndim == 2:  # univariate CSC
        ds_init = np.swapaxes(ds_init, 0, 1)[:, None, :]
        X = np.swapaxes(X, 0, 1)[:, None, :]
        single_channel = True
    else:  # multivariate CSC
        ds_init = np.swapaxes(ds_init, 0, 2)
        X = np.swapaxes(X, 0, 2)
        single_channel = False

    options = {
        'Verbose': verbose > 0,
        'StatusHeader': False,
        'MaxMainIter': n_iter,
        'CBPDN': dict(NonNegCoef=True),
        'CCMOD': dict(ZeroMean=False),
        'DictSize': ds_init.shape,
    }

    # wolberg / convolutional basis pursuit
    opt = ConvBPDNDictLearn.Options(options)
    cbpdn = ConvBPDNDictLearn(ds_init, X, reg, opt, dimN=1)
    results = cbpdn.solve()
    times = np.cumsum(cbpdn.getitstat().Time)

    d_hat, pobj = results
    if single_channel:  # univariate CSC
        d_hat = d_hat.squeeze().T
        n_atoms, n_times_atom = d_hat.shape
    else:
        d_hat = d_hat.squeeze().swapaxes(0, 2)
        n_atoms, n_channels, n_times_atom = d_hat.shape

    z_hat = cbpdn.getcoef().squeeze().swapaxes(0, 2)
    times = np.concatenate([[0], times])

    # z_hat.shape = (n_atoms, n_trials, n_times)
    z_hat = z_hat[:, :, :-n_times_atom + 1]
    # z_hat.shape = (n_atoms, n_trials, n_times_valid)

    return pobj, times, d_hat, z_hat


def run_fista(X, ds_init, reg, n_iter, random_state, label):
    assert X.ndim == 2
    n_atoms, n_times_atom = ds_init.shape
    pobj, times, d_hat, z_hat = learn_d_z(
        X, n_atoms, n_times_atom, func_d=update_d_block, solver_z='fista',
        solver_z_kwargs=dict(max_iter=2), reg=reg, n_iter=n_iter,
        random_state=random_state, ds_init=ds_init, n_jobs=1, verbose=verbose)

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


def run_l_bfgs(X, ds_init, reg, n_iter, random_state, label, factr_d=1e7,
               factr_z=1e14):
    assert X.ndim == 2
    n_atoms, n_times_atom = ds_init.shape
    pobj, times, d_hat, z_hat = learn_d_z(
        X, n_atoms, n_times_atom,
        func_d=update_d_block, solver_z='l-bfgs', solver_z_kwargs=dict(
            factr=factr_z), reg=reg, n_iter=n_iter, solver_d_kwargs=dict(
                factr=factr_d), random_state=random_state, ds_init=ds_init,
        n_jobs=1, verbose=verbose)

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


def run_multichannel_gcd(X, ds_init, reg, n_iter, random_state, label):
    if X.ndim == 2:
        n_atoms, n_times_atom = ds_init.shape
        ds_init = np.c_[np.ones((n_atoms, 1)), ds_init]
        X = X[:, None, :]
    else:
        n_atoms, n_channels, n_times_atom = ds_init.shape
        ds_init = get_uv(ds_init)  # project init to rank 1

    solver_z_kwargs = dict(max_iter=2, tol=1e-3)
    pobj, times, d_hat, z_hat, reg = learn_d_z_multi(
        X, n_atoms, n_times_atom, solver_d='alternate_adaptive',
        solver_z="lgcd", uv_constraint='separate', eps=-np.inf,
        solver_z_kwargs=solver_z_kwargs, reg=reg, solver_d_kwargs=dict(
            max_iter=100), n_iter=n_iter, random_state=random_state,
        raise_on_increase=False, D_init=ds_init, n_jobs=1, verbose=verbose)

    # remove the ds init duration
    times[0] = 0

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


def run_multichannel_gcd_fullrank(X, ds_init, reg, n_iter, random_state,
                                  label):
    assert X.ndim == 3
    n_atoms, n_channels, n_times_atom = ds_init.shape

    solver_z_kwargs = dict(max_iter=2, tol=1e-3)
    pobj, times, d_hat, z_hat, reg = learn_d_z_multi(
        X, n_atoms, n_times_atom, solver_d='fista', solver_z="lgcd",
        uv_constraint='separate', eps=-np.inf, solver_z_kwargs=solver_z_kwargs,
        reg=reg, solver_d_kwargs=dict(max_iter=100), n_iter=n_iter,
        random_state=random_state, raise_on_increase=False, D_init=ds_init,
        n_jobs=1, verbose=verbose, rank1=False)

    # remove the ds init duration
    times[0] = 0

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


def colorify(message, color=BLUE):
    """Change color of the standard output"""
    return ("\033[1;%dm" % color) + message + "\033[0m"


#########################################
# List of functions used in the benchmark
#########################################

n_iter = 100
methods_univariate = [
    [run_cbpdn, 'Garcia-Cardona et al (2017)', n_iter * 2],
    [run_fista, 'Jas et al (2017) FISTA', n_iter],
    [run_l_bfgs, 'Jas et al (2017) LBFGS', n_iter],
    [run_multichannel_gcd, 'Proposed (univariate)', n_iter],
]

n_iter_multi = 20
methods_multivariate = [
    [run_cbpdn, 'Wohlberg (2016)', n_iter_multi * 2],
    [run_multichannel_gcd_fullrank, 'Proposed (multivariate)', n_iter_multi],
    [run_multichannel_gcd, 'Proposed (multichannel)', n_iter_multi],
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
        X_init = X[:, None, :]
    else:
        n_trials, n_channels, n_times = X.shape
        X_init = X

    # use the same init for all methods
    ds_init = init_dictionary(X_init, n_atoms, n_times_atom, D_init='chunk',
                              rank1=False, uv_constraint='separate',
                              D_init_params=dict(), random_state=random_state)
    if len(X_shape) == 2:
        ds_init = ds_init[:, 0, :]

    # run the selected algorithm with one iter to remove compilation overhead
    _, _, _, _ = func(X, ds_init, reg, 1, random_state, label)

    # run the selected algorithm
    pobj, times, d_hat, z_hat = func(X, ds_init, reg, n_iter, random_state,
                                     label)

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

    for params in out_iterator:
        n_times_atom, n_atoms, n_channels, reg = params
        msg = 'n_times_atom, n_atoms, n_channels, reg = ' + str(params)
        print(colorify(msg, RED))
        print(colorify('-' * len(msg), RED))

        save_name = base_name + str(params)
        save_name = os.path.join('figures', save_name)

        all_results = []

        X, info = load_data(
            dataset='somato', epoch=False, n_jobs=n_jobs, n_trials=2
        )

        if n_channels == 1:
            X = X[:, 0, :]  # take only one channel
        elif n_channels is not None:
            X = X[:, :n_channels, :]

        assert X.shape[0] > 1  # we need at least two trials for sporco
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

        all_results_df = pd.DataFrame(
            all_results, columns='random_state label pobj times d_hat '
            'z_hat n_atoms n_times_atom n_trials n_times n_channels reg'.
            split(' '))
        all_results_df.to_pickle(save_name + '.pkl')

    print('-- End of the script --')
