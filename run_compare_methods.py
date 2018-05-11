from __future__ import print_function
import os
import itertools
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.externals.joblib import Parallel, delayed, Memory

import alphacsc.other.heide_csc as CSC
from sporco.admm.cbpdndl import ConvBPDNDictLearn

from alphacsc.utils.profile_this import profile_this  # noqa
from alphacsc.update_d import update_d_block
from alphacsc.learn_d_z import learn_d_z
from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.datasets.somato import load_data
from alphacsc.init_dict import init_dictionary
from alphacsc.utils.dictionary import get_uv

mem = Memory(cachedir='.', verbose=0)

START = time.time()

##############################
# Parameters of the simulation

verbose = 1

debug = False
if debug:
    base_name = 'debug_'
    # n_jobs for the parallel running of single core methods
    n_jobs = 1
    # number of random states
    n_states = 1
    # loop over parameters
    n_channel_list = [1]
    n_atoms_list = [2]
    n_times_atom_list = [16]
    reg_list = [0.3]
else:
    base_name = 'run_1'
    # n_jobs for the parallel running of single core methods
    n_jobs = -3
    # number of random states
    n_states = 10
    # loop over parameters
    n_times_atom_list = [16, 64]
    n_atoms_list = [2, 8]
    n_channel_list = [1, 5, 25]
    reg_list = [0.3, 1., 3., 10.]

##############################
# methods


def run_admm(X, ds_init, reg, n_iter, random_state, label, max_it_d=10,
             max_it_z=10):
    # admm with the following differences
    # positivity constraints
    # different init
    # d step and z step are swapped
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
    # use only one thread in fft
    import sporco.linalg
    sporco.linalg.pyfftw_threads = 1

    if X.ndim == 2:
        ds_init = np.swapaxes(ds_init, 0, 1)[:, None, :]
        X = np.swapaxes(X, 0, 1)[:, None, :]
        single_channel = True
    else:
        ds_init = np.swapaxes(ds_init, 0, 2)
        X = np.swapaxes(X, 0, 2)
        single_channel = False

    options = {
        'Verbose': verbose > 0,
        'MaxMainIter': n_iter,
        'CBPDN': dict(rho=50.0 * reg + 0.5, NonNegCoef=True),
        'CCMOD': dict(ZeroMean=False),
        'DictSize': ds_init.shape,
    }

    # wolberg / convolutional basis pursuit
    opt = ConvBPDNDictLearn.Options(options)
    cbpdn = ConvBPDNDictLearn(ds_init, X, reg, opt, dimN=1)
    results = cbpdn.solve()
    times = np.cumsum(cbpdn.getitstat().Time)

    d_hat, pobj = results
    if single_channel:
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


def run_ista(X, ds_init, reg, n_iter, random_state, label):
    assert X.ndim == 2
    n_atoms, n_times_atom = ds_init.shape
    pobj, times, d_hat, z_hat = learn_d_z(
        X, n_atoms, n_times_atom, func_d=update_d_block, solver_z='ista',
        solver_z_kwargs=dict(max_iter=2), reg=reg, n_iter=n_iter,
        random_state=random_state, ds_init=ds_init, n_jobs=1, verbose=verbose)

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


def run_fista(X, ds_init, reg, n_iter, random_state, label):
    assert X.ndim == 2
    n_atoms, n_times_atom = ds_init.shape
    pobj, times, d_hat, z_hat = learn_d_z(
        X, n_atoms, n_times_atom, func_d=update_d_block, solver_z='fista',
        solver_z_kwargs=dict(max_iter=2), reg=reg, n_iter=n_iter,
        random_state=random_state, ds_init=ds_init, n_jobs=1, verbose=verbose)

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


def run_lbfgs(X, ds_init, reg, n_iter, random_state, label, factr_d=1e7,
              factr_z=1e14):
    assert X.ndim == 2
    n_atoms, n_times_atom = ds_init.shape
    pobj, times, d_hat, z_hat = learn_d_z(
        X, n_atoms, n_times_atom,
        func_d=update_d_block, solver_z='l_bfgs', solver_z_kwargs=dict(
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
        ds_init = get_uv(ds_init)  # project init to rank 1

    solver_z_kwargs = dict(max_iter=2, tol=1e-3)
    pobj, times, d_hat, z_hat = learn_d_z_multi(
        X, n_atoms, n_times_atom, solver_d='alternate_adaptive',
        solver_z='gcd', uv_constraint='separate', eps=1e-14,
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
    pobj, times, d_hat, z_hat = learn_d_z_multi(
        X, n_atoms, n_times_atom, solver_d='fista', solver_z='gcd',
        uv_constraint='separate', eps=1e-14, solver_z_kwargs=solver_z_kwargs,
        reg=reg, solver_d_kwargs=dict(max_iter=100), n_iter=n_iter,
        random_state=random_state, raise_on_increase=False, D_init=ds_init,
        n_jobs=1, verbose=verbose, rank1=False)

    # remove the ds init duration
    times[0] = 0

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


def run_multichannel_lbfgs(X, ds_init, reg, n_iter, random_state, label):
    if X.ndim == 2:
        n_atoms, n_times_atom = ds_init.shape
        ds_init = np.c_[np.ones((n_atoms, 1)), ds_init]
        X = X[:, None, :]
    else:
        n_atoms, n_channels, n_times_atom = ds_init.shape
        ds_init = get_uv(ds_init)  # project init to rank 1

    pobj, times, d_hat, z_hat = learn_d_z_multi(
        X, n_atoms, n_times_atom, solver_d='alternate_adaptive',
        uv_constraint='separate', solver_z_kwargs=dict(
            factr=1e15), eps=1e-14, reg=reg, solver_d_kwargs=dict(
                max_iter=100), n_iter=n_iter, random_state=random_state,
        raise_on_increase=False, D_init=ds_init, n_jobs=1, verbose=verbose)

    # remove the ds init duration
    times[0] = 0

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


def run_multichannel_gcd_sparse(X, ds_init, reg, n_iter, random_state, label):
    if X.ndim == 2:
        n_atoms, n_times_atom = ds_init.shape
        ds_init = np.c_[np.ones((n_atoms, 1)), ds_init]
        X = X[:, None, :]
    else:
        n_atoms, n_channels, n_times_atom = ds_init.shape
        ds_init = get_uv(ds_init)  # project init to rank 1

    solver_z_kwargs = dict(max_iter=2, tol=1e-3)
    pobj, times, d_hat, z_hat = learn_d_z_multi(
        X, n_atoms, n_times_atom, solver_d='alternate_adaptive',
        uv_constraint='separate', solver_z='gcd', eps=1e-14,
        solver_z_kwargs=solver_z_kwargs, reg=reg, solver_d_kwargs=dict(
            max_iter=100), use_sparse_z=True, n_iter=n_iter,
        raise_on_increase=False, random_state=random_state, D_init=ds_init,
        n_jobs=1, verbose=verbose)

    # remove the ds init duration
    times[0] = 0

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


n_iter = 1000
methods_univariate = [
    # [run_multichannel_alt_lbfgs, 'find_best_pobj', n_iter * 5],
    # [run_admm, 'Heide & al (2015)', n_iter // 2],  # FIXME: going up
    [run_cbpdn, 'Wohlberg (2017)', n_iter * 5],
    # [run_ista, 'Jas & al (2017) ISTA', n_iter * 3]],
    [run_fista, 'Jas & al (2017) FISTA', n_iter],
    [run_lbfgs, 'Jas & al (2017) LBFGS', n_iter],
    # [run_multichannel_lbfgs, 'multiCSC LBFGS', n_iter],
    [run_multichannel_gcd, 'LGCD (1 channel)', n_iter],
    # [run_multichannel_gcd_sparse, 'multiCSC LGCD sparse', n_iter],
]

n_iter_multi = 200
methods_multivariate = [
    [run_cbpdn, 'Wohlberg (2016)', n_iter_multi * 5],
    [run_multichannel_gcd_fullrank, 'LGCD (full rank)', n_iter_multi],
    [run_multichannel_gcd, 'LGCD (rank 1)', n_iter_multi],
]

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(30, 38)


def colorify(message, color=BLUE):
    return ("\033[1;%dm" % color) + message + "\033[0m"


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
        reg = reg * n_channels

    # use the same init for all methods
    ds_init = init_dictionary(X_init, n_atoms, n_times_atom, D_init='chunk',
                              rank1=False, uv_constraint='separate',
                              kmeans_params=dict(), random_state=random_state)
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


if __name__ == '__main__':

    # cached_one_run = mem.cache(func=one_run, ignore=['X'])

    out_iterator = itertools.product(n_times_atom_list, n_atoms_list,
                                     n_channel_list, reg_list)

    for params in out_iterator:
        try:
            n_times_atom, n_atoms, n_channels, reg = params
            msg = 'n_times_atom, n_atoms, n_channels, reg = ' + str(params)
            print(colorify(msg, RED))
            print(colorify('-' * len(msg), RED))

            save_name = base_name + str(params)
            save_name = os.path.join('figures', save_name)

            all_results = []

            X, info = load_data(epoch=False, n_jobs=n_jobs, n_trials=2)

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
        except Exception as e:
            print(e)
            pass

    print('-- End of the script --')
