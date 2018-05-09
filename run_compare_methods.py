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
from alphacsc.utils import check_random_state
from alphacsc.update_d import update_d_block
from alphacsc.learn_d_z import learn_d_z
from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.datasets.somato import load_data

mem = Memory(cachedir='.', verbose=0)

START = time.time()

##############################
# Parameters of the simulation
verbose = 1

# n_jobs for the parallel running of single core methods
n_jobs = 5
# number of random states
n_states = 1

n_times_atom = 129  # L
assert n_times_atom % 2 == 1  # has to be odd for [Heide et al 2016]
n_atoms = 2  # K
reg = 1.

save_name = 'methods_'
save_name = os.path.join('figures', save_name)


def run_admm(X, ds_init, reg, n_iter, random_state, label,
             max_it_d=10, max_it_z=10):
    # admm with the following differences
    # positivity constraints
    # different init
    # d step and z step are swapped
    tol = np.float64(1e-3)
    size_kernel = ds_init.shape
    assert size_kernel[1] % 2 == 1
    [d, z, Dz, list_obj_val, times_admm] = CSC.learn_conv_sparse_coder(
        X, size_kernel, max_it=n_iter, tol=tol, random_state=random_state,
        lambda_prior=reg, ds_init=ds_init, verbose=verbose,
        max_it_d=max_it_d, max_it_z=max_it_z)

    # z.shape = (n_trials, n_atoms, n_times + 2 * n_times_atom)
    z = z[:, :, 2 * n_times_atom:-2 * n_times_atom]
    z = z.swapaxes(0, 1)
    # z.shape = (n_atoms, n_trials, n_times - 2 * n_times_atom)

    return list_obj_val, np.cumsum(times_admm)[::2], d, z


def run_cbpdn(X, ds_init, reg, n_iter, random_state, label):
    # use only one thread in fft
    import sporco.linalg
    sporco.linalg.pyfftw_threads = 1

    # wolberg / convolutional basis pursuit
    opt = ConvBPDNDictLearn.Options({
        'Verbose': verbose > 0,
        'MaxMainIter': n_iter,
        'CBPDN': dict(rho=50.0 * reg + 0.5, NonNegCoef=True),
        'CCMOD': dict(ZeroMean=False),
    })
    cbpdn = ConvBPDNDictLearn(
        np.swapaxes(ds_init, 0, 1)[:, None, :],
        np.swapaxes(X, 0, 1)[:, None, :], reg, opt)
    results = cbpdn.solve()
    times = np.cumsum(cbpdn.getitstat().Time)

    d_hat, pobj = results
    d_hat = d_hat.squeeze().T
    z_hat = cbpdn.getcoef().squeeze().swapaxes(0, 2)
    times = np.concatenate([[0], times])

    n_atoms, n_times_atom = ds_init.shape
    # z_hat.shape = (n_atoms, n_trials, n_times)
    z_hat = z_hat[:, :, :-n_times_atom + 1]
    # z_hat.shape = (n_atoms, n_trials, n_times_valid)

    return pobj, times, d_hat, z_hat


def run_ista(X, ds_init, reg, n_iter, random_state, label):
    n_atoms, n_times_atom = ds_init.shape
    pobj, times, d_hat, z_hat = learn_d_z(
        X, n_atoms, n_times_atom, func_d=update_d_block, solver_z='ista',
        solver_z_kwargs=dict(max_iter=5), reg=reg, n_iter=n_iter,
        random_state=random_state, ds_init=ds_init, n_jobs=1,
        verbose=verbose)

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


def run_fista(X, ds_init, reg, n_iter, random_state, label):
    n_atoms, n_times_atom = ds_init.shape
    pobj, times, d_hat, z_hat = learn_d_z(
        X, n_atoms, n_times_atom, func_d=update_d_block, solver_z='fista',
        solver_z_kwargs=dict(max_iter=5), reg=reg, n_iter=n_iter,
        random_state=random_state, ds_init=ds_init, n_jobs=1,
        verbose=verbose)

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


def run_lbfgs(X, ds_init, reg, n_iter, random_state, label, factr_d=1e7,
              factr_z=1e14):
    n_atoms, n_times_atom = ds_init.shape
    pobj, times, d_hat, z_hat = learn_d_z(
        X, n_atoms, n_times_atom,
        func_d=update_d_block, solver_z='l_bfgs', solver_z_kwargs=dict(
            factr=factr_z), reg=reg, n_iter=n_iter, solver_d_kwargs=dict(
                factr=factr_d), random_state=random_state, ds_init=ds_init,
        n_jobs=1, verbose=verbose)

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


# @profile_this
def run_multichannel_alt_gcd(X, ds_init, reg, n_iter, random_state, label):
    n_atoms, n_times_atom = ds_init.shape
    D_init = np.c_[np.ones((n_atoms, 1)), ds_init]

    solver_z_kwargs = dict(max_iter=500, tol=1e-1)
    pobj, times, d_hat, z_hat = learn_d_z_multi(
        X[:, None, :], n_atoms, n_times_atom, solver_d='alternate_adaptive',
        solver_z='gcd', uv_constraint='separate', eps=1e-14,
        solver_z_kwargs=solver_z_kwargs, reg=reg, solver_d_kwargs=dict(
            max_iter=100), n_iter=n_iter, random_state=random_state,
        D_init=D_init, n_jobs=1, verbose=verbose)

    # remove the ds init duration
    times[0] = 0

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


def run_multichannel_alt_lbfgs(X, ds_init, reg, n_iter, random_state, label):
    n_atoms, n_times_atom = ds_init.shape
    D_init = np.c_[np.ones((n_atoms, 1)), ds_init]
    pobj, times, d_hat, z_hat = learn_d_z_multi(
        X[:, None, :], n_atoms, n_times_atom, solver_d='alternate_adaptive',
        uv_constraint='separate', solver_z_kwargs=dict(
            factr=1e15), eps=1e-14, reg=reg, solver_d_kwargs=dict(
                max_iter=100), n_iter=n_iter, random_state=random_state,
        D_init=D_init, n_jobs=1, verbose=verbose)

    # remove the ds init duration
    times[0] = 0

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


# @profile_this
def run_multichannel_alt_gcd_sparse(X, ds_init, reg, n_iter, random_state,
                                    label):
    n_atoms, n_times_atom = ds_init.shape
    D_init = np.c_[np.ones((n_atoms, 1)), ds_init]

    solver_z_kwargs = dict(max_iter=500, tol=1e-1)
    pobj, times, d_hat, z_hat = learn_d_z_multi(
        X[:, None, :], n_atoms, n_times_atom, solver_d='alternate_adaptive',
        uv_constraint='separate', solver_z='gcd', eps=1e-14,
        solver_z_kwargs=solver_z_kwargs, reg=reg, solver_d_kwargs=dict(
            max_iter=40), use_sparse_z=True, n_iter=n_iter,
        random_state=random_state, D_init=D_init, n_jobs=1,
        verbose=verbose)

    # remove the ds init duration
    times[0] = 0

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


n_iter = 200
methods = [
    # [run_multichannel_alt_lbfgs, 'find_best_pobj', n_iter * 5],
    # [run_admm, 'Heide & al (2015)', n_iter // 2],  # FIXME
    [run_cbpdn, 'Wohlberg (2017)', n_iter * 4],
    # [run_ista, 'Jas & al (2017) ISTA', n_iter * 2]],
    [run_fista, 'Jas & al (2017) FISTA', n_iter * 2],
    [run_lbfgs, 'Jas & al (2017) LBFGS', n_iter * 2],
    [run_multichannel_alt_lbfgs, 'multiCSC LBFGS', n_iter],
    [run_multichannel_alt_gcd, 'multiCSC LGCD', n_iter],
    [run_multichannel_alt_gcd_sparse, 'multiCSC LGCD sparse', n_iter],
]


BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(30, 38)


def colorify(message, color=BLUE):
    return ("\033[1;%dm" % color) + message + "\033[0m"


def one_run(X, X_shape, random_state, method, n_atoms, n_times_atom, reg=reg):
    n_trials, n_times = X.shape
    func, label, n_iter = method
    current_time = time.time() - START
    msg = ('%s - %s: started at T=%.0f sec'
           % (random_state, label, current_time))
    print(colorify(msg, BLUE))

    # use the same init for all methods
    init = 'chunk'
    rng = check_random_state(random_state)
    if init == 'random':
        ds_init = rng.randn(n_atoms, n_times_atom)
    elif init == 'chunk':
        ds_init = np.zeros((n_atoms, n_times_atom))
        for i_atom in range(n_atoms):
            i_trial = rng.randint(n_trials)
            t0 = rng.randint(n_times - n_times_atom)
            ds_init[i_atom] = X[i_trial, t0:t0 + n_times_atom]
    else:
        raise ValueError()

    d_norm = np.linalg.norm(ds_init, axis=1)
    ds_init /= d_norm[:, None]

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
    msg = ('%s - %s: done in %.0f sec at T=%.0f sec'
           % (random_state, label, duration, current_time))
    print(colorify(msg, GREEN))
    return (random_state, label, np.asarray(pobj), np.asarray(times),
            np.asarray(d_hat), np.asarray(z_hat), n_atoms, n_times_atom,
            n_trials, n_times, reg)


if __name__ == '__main__':

    cached_one_run = mem.cache(func=one_run, ignore=['X'])

    all_results = []
    print(n_atoms, n_times_atom)

    X, info = load_data(epoch=False, n_jobs=n_jobs, n_trials=2)
    X = X[:, 0, :]  # take only one channel
    assert X.shape[0] > 1  # we need at least two trials for sporco
    X_shape = X.shape

    iterator = itertools.product(methods, range(n_states))
    if n_jobs == 1:
        results = [
            one_run(X, X_shape, random_state, method, n_atoms, n_times_atom)
            for method, random_state in iterator
        ]
    else:
        # run the methods for different random_state
        delayed_one_run = delayed(cached_one_run)
        results = Parallel(n_jobs=n_jobs)(
            delayed_one_run(X, X_shape, random_state, method, n_atoms,
                            n_times_atom)
            for method, random_state in iterator)

    all_results.extend(results)

    # save even intermediate results
    all_results_df = pd.DataFrame(
        all_results, columns='random_state label pobj times d_hat '
        'z_hat n_atoms n_times_atom n_trials n_times reg'.split(' '))
    all_results_df.to_pickle(save_name + '.pkl')

    print('-- End of the script --')
