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
n_jobs = 6
# number of random states
n_states = 1

n_times_atom = 128  # L
n_atoms = 2  # K
reg = 5.0

# A method stops if its objective function reaches best_pobj * (1 + threshold)
threshold = -1

save_name = 'methods_'
save_name = os.path.join('figures', save_name)


###################
# simulate the data
@mem.cache()
def load(n_atoms, n_times_atom, reg):
    X, info = load_data(epoch=False, n_jobs=n_jobs)
    X = X[:, 0, :]

    solver_z_kwargs = dict(max_iter=500, tol=1e-4)
    print('Finding best_pobj...')
    rng = check_random_state(0)
    ds_init = rng.randn(n_atoms, n_times_atom)
    D_init = np.c_[np.ones((n_atoms, 1)), ds_init]

    pobj, _, _, _ = learn_d_z_multi(
        X[:, None, :], n_atoms, n_times_atom, solver_d='alternate_adaptive',
        solver_z='l_bfgs', uv_constraint='separate',
        solver_z_kwargs=solver_z_kwargs, reg=reg, solver_d_kwargs=dict(
            max_iter=50), n_iter=1000, random_state=0,
        D_init=D_init, n_jobs=n_jobs, stopping_pobj=None, verbose=verbose)
    best_pobj = pobj[-1]
    print('[Done]')
    return X, best_pobj


def run_admm(X, ds_init, reg, n_iter, random_state, label, stopping_pobj,
             max_it_d=10, max_it_z=10):
    # admm with the following differences
    # positivity constraints
    # different init
    # d step and z step are swapped
    tol = np.float64(1e-3)
    size_kernel = ds_init.shape
    [d, z, Dz, list_obj_val, times_admm] = CSC.learn_conv_sparse_coder(
        X, size_kernel, max_it=n_iter, tol=tol, random_state=random_state,
        lambda_prior=reg, ds_init=ds_init, verbose=verbose,
        stopping_pobj=stopping_pobj, max_it_d=max_it_d, max_it_z=max_it_z)

    # z.shape = (n_trials, n_atoms, n_times + 2 * n_times_atom)
    z = z[:, :, 2 * n_times_atom:-2 * n_times_atom]
    z = z.swapaxes(0, 1)
    # z.shape = (n_atoms, n_trials, n_times - 2 * n_times_atom)

    return list_obj_val, np.cumsum(times_admm)[::2], d, z


def run_cbpdn(X, ds_init, reg, n_iter, random_state, label, stopping_pobj):
    # wolberg / convolutional basis pursuit
    opt = ConvBPDNDictLearn.Options({
        'Verbose': verbose > 0,
        'MaxMainIter': n_iter,
        'CBPDN': dict(rho=50.0 * reg + 0.5, NonNegCoef=True),
        'CCMOD': dict(ZeroMean=False),
    })
    cbpdn = ConvBPDNDictLearn(
        np.swapaxes(ds_init, 0, 1)[:, None, :],
        np.swapaxes(X, 0, 1)[:, None, :], reg, opt,
        stopping_pobj=stopping_pobj)
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


def run_ista(X, ds_init, reg, n_iter, random_state, label, stopping_pobj):
    n_atoms, n_times_atom = ds_init.shape
    pobj, times, d_hat, z_hat = learn_d_z(
        X, n_atoms, n_times_atom, func_d=update_d_block, solver_z='ista',
        solver_z_kwargs=dict(max_iter=5), reg=reg, n_iter=n_iter,
        random_state=random_state, ds_init=ds_init, n_jobs=1,
        stopping_pobj=stopping_pobj, verbose=verbose)

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


def run_fista(X, ds_init, reg, n_iter, random_state, label, stopping_pobj):
    n_atoms, n_times_atom = ds_init.shape
    pobj, times, d_hat, z_hat = learn_d_z(
        X, n_atoms, n_times_atom, func_d=update_d_block, solver_z='fista',
        solver_z_kwargs=dict(max_iter=5), reg=reg, n_iter=n_iter,
        random_state=random_state, ds_init=ds_init, n_jobs=1,
        stopping_pobj=stopping_pobj, verbose=verbose)

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


def run_lbfgs(X, ds_init, reg, n_iter, random_state, label, stopping_pobj,
              factr_d=1e7, factr_z=1e14):
    n_atoms, n_times_atom = ds_init.shape
    pobj, times, d_hat, z_hat = learn_d_z(
        X, n_atoms, n_times_atom,
        func_d=update_d_block, solver_z='l_bfgs', solver_z_kwargs=dict(
            factr=factr_z), reg=reg, n_iter=n_iter, solver_d_kwargs=dict(
                factr=factr_d), random_state=random_state, ds_init=ds_init,
        n_jobs=1, stopping_pobj=stopping_pobj, verbose=verbose)

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


# @profile_this
def run_multichannel_alt_gcd(X, ds_init, reg, n_iter, random_state, label,
                             stopping_pobj):
    n_atoms, n_times_atom = ds_init.shape
    D_init = np.c_[np.ones((n_atoms, 1)), ds_init]

    solver_z_kwargs = dict(max_iter=500, tol=2e-1)
    pobj, times, d_hat, z_hat = learn_d_z_multi(
        X[:, None, :], n_atoms, n_times_atom, solver_d='alternate_adaptive',
        solver_z='gcd', uv_constraint='separate',
        solver_z_kwargs=solver_z_kwargs, reg=reg, solver_d_kwargs=dict(
            max_iter=40), n_iter=n_iter, random_state=random_state,
        D_init=D_init, n_jobs=1, stopping_pobj=stopping_pobj, verbose=verbose)

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


def run_multichannel_alt_lbfgs(X, ds_init, reg, n_iter, random_state, label,
                               stopping_pobj):
    n_atoms, n_times_atom = ds_init.shape
    D_init = np.c_[np.ones((n_atoms, 1)), ds_init]
    pobj, times, d_hat, z_hat = learn_d_z_multi(
        X[:, None, :], n_atoms, n_times_atom, solver_d='alternate_adaptive',
        uv_constraint='separate', solver_z_kwargs=dict(
            factr=8e14), reg=reg, solver_d_kwargs=dict(
                max_iter=40), n_iter=n_iter, random_state=random_state,
        D_init=D_init, n_jobs=1, stopping_pobj=stopping_pobj, verbose=verbose)

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


# @profile_this
def run_multichannel_alt_gcd_sparse(X, ds_init, reg, n_iter, random_state,
                                    label, stopping_pobj):
    n_atoms, n_times_atom = ds_init.shape
    D_init = np.c_[np.ones((n_atoms, 1)), ds_init]

    solver_z_kwargs = dict(max_iter=500, tol=2e-1)
    pobj, times, d_hat, z_hat = learn_d_z_multi(
        X[:, None, :], n_atoms, n_times_atom, solver_d='alternate_adaptive',
        uv_constraint='separate', solver_z='gcd',
        solver_z_kwargs=solver_z_kwargs, reg=reg, solver_d_kwargs=dict(
            max_iter=40), use_sparse_z=True, n_iter=n_iter,
        random_state=random_state, D_init=D_init, n_jobs=1,
        stopping_pobj=stopping_pobj, verbose=verbose)

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


n_iter = 1000
methods = [
    [run_admm, 'Heide et al (2015)', n_iter // 10],
    [run_cbpdn, 'Wohlberg (2017)', n_iter],
    [run_ista, 'Jas et al (2017) ista', n_iter],
    [run_fista, 'Jas et al (2017) fista', n_iter],
    [run_lbfgs, 'Jas et al (2017) lbfgs', n_iter],
    [run_multichannel_alt_lbfgs, 'multiCSC_lbfgs', n_iter],
    [run_multichannel_alt_gcd, 'multiCSC_gcd', n_iter // 5],
    [run_multichannel_alt_gcd_sparse, 'multiCSC_gcd_sparse', n_iter // 5],
]


BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(30, 38)


def colorify(message, color=BLUE):
    return ("\033[1;%dm" % color) + message + "\033[0m"


def one_run(X, X_shape, random_state, method, n_atoms, n_times_atom,
            stopping_pobj, best_pobj, reg=reg):
    n_trials, n_times = X_shape
    func, label, n_iter = method
    current_time = time.time() - START
    msg = ('%s - %s: started at T=%.0f sec'
           % (random_state, label, current_time))
    print(colorify(msg, BLUE))

    # use the same init for all methods
    rng = check_random_state(random_state)
    ds_init = rng.randn(n_atoms, n_times_atom)

    # run the selected algorithm with one iter to remove compilation overhead
    _, _, _, _ = func(X, ds_init, reg, 1, random_state, label, stopping_pobj)

    # run the selected algorithm
    pobj, times, d_hat, z_hat = func(X, ds_init, reg, n_iter, random_state,
                                     label, stopping_pobj)

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
            n_trials, n_times, stopping_pobj, best_pobj)


if __name__ == '__main__':

    cached_one_run = mem.cache(func=one_run, ignore=['X'])

    all_results = []
    print(n_atoms, n_times_atom)
    # simulate the data and optimize to get the best pobj
    X, best_pobj = load(n_atoms, n_times_atom, reg)
    X_shape = X.shape
    stopping_pobj = best_pobj * (1 + threshold)

    iterator = itertools.product(methods, range(n_states))
    if n_jobs == 1:
        results = [
            cached_one_run(X, X_shape, random_state, method, n_atoms,
                           n_times_atom, stopping_pobj, best_pobj)
            for method, random_state in iterator
        ]
    else:
        # run the methods for different random_state
        delayed_one_run = delayed(cached_one_run)
        results = Parallel(n_jobs=n_jobs)(
            delayed_one_run(X, X_shape, random_state, method, n_atoms,
                            n_times_atom, stopping_pobj, best_pobj)
            for method, random_state in iterator)

    # # add the multicore runs outside the parallel loop
    # if methods[-1][0] is not None:
    #     for random_state in range(n_states):
    #         results.append(
    #             one_run(
    #                 X, X_shape, random_state, methods[-1], n_atoms,
    #                 n_times_atom, stopping_pobj, best_pobj))

    all_results.extend(results)

    # save even intermediate results
    all_results_df = pd.DataFrame(
        all_results, columns='random_state label pobj times d_hat '
        'z_hat n_atoms n_times_atom n_trials n_times '
        'stopping_pobj best_pobj'.split(' '))
    all_results_df.to_pickle(save_name + '.pkl')

    print('-- End of the script --')
