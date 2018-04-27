from __future__ import print_function
import os
import itertools
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.externals.joblib import Parallel, delayed, Memory

from alphacsc.utils.profile_this import profile_this  # noqa
from alphacsc.utils import check_random_state, get_D
from alphacsc.learn_d_z_multi import learn_d_z_multi

mem = Memory(cachedir='.', verbose=0)

START = time.time()

##############################
# Parameters of the simulation
verbose = 1

# n_jobs for the parallel running of single core methods
n_jobs = 50
# number of random states
n_states = 10

n_trials = 10  # N
n_times_atom = 128  # L
n_times = 20000  # T
n_atoms = 2  # K
reg = 1.0

# A method stops if its objective function reaches best_pobj * (1 + threshold)
threshold = -1

save_name = 'methods_scaling.pkl'
if not os.path.exists("figures"):
    os.mkdir("figures")
save_name = os.path.join('figures', save_name)


def generate_D_init(n_channels, random_state):
    rng = check_random_state(random_state)
    return rng.randn(n_atoms, n_channels + n_times_atom)


def find_best_pobj(X, n_channels):

    solver_z_kwargs = dict(max_iter=500, tol=1e-2)
    D_init = generate_D_init(n_channels)
    print('Finding best_pobj for n_channels={}...'.format(n_channels))
    pobj, _, _, _ = learn_d_z_multi(
        X[:, :n_channels, :], n_atoms, n_times_atom,
        solver_z='l_bfgs', solver_z_kwargs=solver_z_kwargs,
        solver_d='alternate_adaptive', solver_d_kwargs=dict(max_iter=50),
        uv_constraint='separate', reg=reg, n_iter=10, random_state=0,
        D_init=D_init, n_jobs=1, stopping_pobj=None, verbose=verbose)
    best_pobj = pobj[-1]
    print('[Done] n_channels={}'.format(n_channels))
    return n_channels, best_pobj


# @profile_this
def run_multichannel(X, D_init, reg, n_iter, random_state,
                     label, n_channels, stopping_pobj):
    n_atoms, n_channels_n_times_atom = D_init.shape
    n_times_atom = n_channels_n_times_atom - n_channels

    solver_z_kwargs = dict(max_iter=500, tol=1e-1)
    return learn_d_z_multi(
        X, n_atoms, n_times_atom, reg=reg, n_iter=n_iter,
        uv_constraint='separate', rank1=True, D_init=D_init,
        solver_d='alternate_adaptive', solver_d_kwargs=dict(max_iter=50),
        solver_z='gcd', solver_z_kwargs=solver_z_kwargs, use_sparse_z=True,
        stopping_pobj=stopping_pobj,
        name="rank1-{}-{}".format(n_channels, random_state),
        random_state=random_state, n_jobs=1, verbose=verbose)


# @profile_this
def run_multivariate(X, D_init, reg, n_iter, random_state,
                     label, n_channels, stopping_pobj):
    n_atoms, n_channels_n_times_atom = D_init.shape
    n_times_atom = n_channels_n_times_atom - n_channels
    D_init = get_D(D_init, n_channels)

    solver_z_kwargs = dict(max_iter=500, tol=1e-1)
    return learn_d_z_multi(
        X, n_atoms, n_times_atom, reg=reg, n_iter=n_iter,
        uv_constraint='separate', rank1=False, D_init=D_init,
        solver_d='lbfgs', solver_d_kwargs=dict(max_iter=50),
        solver_z='gcd', solver_z_kwargs=solver_z_kwargs, use_sparse_z=True,
        stopping_pobj=stopping_pobj,
        name="dense-{}-{}".format(n_channels, random_state),
        random_state=random_state, n_jobs=1, verbose=verbose)


n_iter = 50
methods = [
    [run_multichannel, 'rank1', n_iter],
    # [run_multivariate, 'dense', n_iter],
]


def one_run(X, n_channels, method, n_atoms, n_times_atom, random_state,
            stopping_pobj, best_pobj, reg=reg):
    func, label, n_iter = method
    current_time = time.time() - START
    print('{}-{}-{}: started at {:.0f} sec'.format(
          label, n_channels, random_state, current_time))

    # use the same init for all methods
    D_init = generate_D_init(n_channels, random_state)
    X = X[:, :n_channels]

    # run the selected algorithm with one iter to remove compilation overhead
    _, _, _, _ = func(X, D_init, reg, 1, random_state, label, n_channels,
                      stopping_pobj)

    # run the selected algorithm
    pobj, times, d_hat, z_hat = func(X, D_init, reg, n_iter, random_state,
                                     label, n_channels, stopping_pobj)

    # store z_hat in a sparse matrix to reduce size
    for z in z_hat:
        z[z < 1e-3] = 0
    z_hat = [sp.csr_matrix(z) for z in z_hat]

    current_time = time.time() - START
    print('{}-{}-{}: done at {:.0f} sec'.format(
          label, n_channels, random_state, current_time))
    assert len(times) > 15
    return (n_channels, random_state, label, np.asarray(pobj),
            np.asarray(times), np.asarray(d_hat), np.asarray(z_hat), n_atoms,
            n_times_atom, n_trials, n_times, stopping_pobj, best_pobj)


if __name__ == '__main__':

    cached_one_run = mem.cache(func=one_run)
    cached_best_pobj = mem.cache(func=find_best_pobj)

    all_results = []
    # load somato data
    from alphacsc.datasets.somato import load_data
    X, info = load_data(epoch=False, n_jobs=n_jobs)

    n_channels = X.shape[1]
    span_channels = np.linspace(1, n_channels, 20).astype(int)

    with Parallel(n_jobs=n_jobs) as parallel:
        # Finding the best_pobj for each n_channels
        delayed_best_pobj = delayed(cached_best_pobj)
        best_pobjs = parallel(delayed_best_pobj(X, n_chan)
                              for n_chan in span_channels)

        stopping_pobj = [
            (n_channels, chan_best_pobj * (1 + threshold), chan_best_pobj)
            for n_channels, chan_best_pobj in best_pobjs]

        iterator = itertools.product(methods, stopping_pobj, range(n_states))
        # run the methods for different random_state
        delayed_one_run = delayed(cached_one_run)
        results = parallel(
            delayed_one_run(X, n_chan, method, n_atoms,
                            n_times_atom, rst, stopping_pobj,
                            best_pobj)
            for method, (n_chan, stopping_pobj, best_pobj), rst in iterator)

        all_results.extend(results)

    # save even intermediate results
    all_results_df = pd.DataFrame(
        all_results, columns='n_channels random_state label pobj times '
        'd_hat z_hat n_atoms n_times_atom n_trials n_times '
        'stopping_pobj best_pobj'.split(' '))
    all_results_df.to_pickle(save_name)
    import IPython
    IPython.embed()

    print('-- End of the script --')
