from __future__ import print_function
import os
import itertools
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.externals.joblib import Parallel, delayed, Memory
from sporco.admm.cbpdndl import ConvBPDNDictLearn

from alphacsc.utils.profile_this import profile_this  # noqa
from alphacsc.utils import check_random_state, get_D
from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.utils.dictionary import get_lambda_max

mem = Memory(cachedir='.', verbose=0)

START = time.time()

##############################
# Parameters of the simulation
verbose = 1

n_trials = 10  # N
n_times_atom = 128  # L
n_times = 20000  # T
n_atoms = 2  # K

save_name = 'methods_scaling_reg{}{}.pkl'
if not os.path.exists("figures"):
    os.mkdir("figures")
save_name = os.path.join('figures', save_name)


def generate_D_init(n_channels, n_times_atom, random_state):
    rng = check_random_state(random_state)
    return rng.randn(n_atoms, n_channels + n_times_atom)


# @profile_this
def run_multichannel(X, D_init, reg, n_iter, random_state,
                     label, n_channels):
    n_atoms, n_channels_n_times_atom = D_init.shape
    n_times_atom = n_channels_n_times_atom - n_channels

    solver_z_kwargs = dict(max_iter=500, tol=1e-1)
    return learn_d_z_multi(
        X, n_atoms, n_times_atom, reg=reg, n_iter=n_iter,
        uv_constraint='separate', rank1=True, D_init=D_init,
        solver_d='alternate_adaptive', solver_d_kwargs=dict(max_iter=50),
        solver_z='gcd', solver_z_kwargs=solver_z_kwargs, use_sparse_z=False,
        name="rank1-{}-{}".format(n_channels, random_state),
        random_state=random_state, n_jobs=1, verbose=verbose)


# @profile_this
def run_multivariate(X, D_init, reg, n_iter, random_state,
                     label, n_channels):
    n_atoms, n_channels_n_times_atom = D_init.shape
    n_times_atom = n_channels_n_times_atom - n_channels
    D_init = get_D(D_init, n_channels)

    solver_z_kwargs = dict(max_iter=500, tol=1e-1)
    return learn_d_z_multi(
        X, n_atoms, n_times_atom, reg=reg, n_iter=n_iter,
        uv_constraint='separate', rank1=False, D_init=D_init,
        solver_d='lbfgs', solver_d_kwargs=dict(max_iter=50),
        solver_z='gcd', solver_z_kwargs=solver_z_kwargs, use_sparse_z=False,
        name="dense-{}-{}".format(n_channels, random_state),
        random_state=random_state, n_jobs=1, verbose=verbose,
        raise_on_increase=False)


# @profile_this
def run_cbpdn(X, ds_init, reg, n_iter, random_state, label, n_channels):
    # use only one thread in fft
    import sporco.linalg
    sporco.linalg.pyfftw_threads = 1

    n_atoms, n_channels_n_times_atom = ds_init.shape
    n_times_atom = n_channels_n_times_atom - n_channels
    ds_init = get_D(ds_init, n_channels)

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
        'CBPDN': dict(NonNegCoef=True),
        'CCMOD': dict(ZeroMean=False),
        'DictSize': ds_init.shape,
    }

    # wohlberg / convolutional basis pursuit
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


def one_run(X, n_channels, method, n_atoms, n_times_atom, random_state, reg):
    func, label, n_iter = method
    current_time = time.time() - START
    print('{}-{}-{}: started at {:.0f} sec'.format(
          label, n_channels, random_state, current_time))

    # use the same init for all methods
    D_init = generate_D_init(n_channels, n_times_atom, random_state)
    X = X[:, :n_channels]

    lmbd_max = get_lambda_max(X, D_init).mean()
    reg_ = reg * lmbd_max

    # run the selected algorithm with one iter to remove compilation overhead
    _, _, _, _ = func(X, D_init, reg_, 1, random_state, label, n_channels)

    # run the selected algorithm
    pobj, times, d_hat, z_hat = func(X, D_init, reg_, n_iter, random_state,
                                     label, n_channels)

    # store z_hat in a sparse matrix to reduce size
    for z in z_hat:
        z[z < 1e-3] = 0
    z_hat = [sp.csr_matrix(z) for z in z_hat]

    current_time = time.time() - START
    print('{}-{}-{}: done at {:.0f} sec'.format(
          label, n_channels, random_state, current_time))
    assert len(times) > 5
    return (n_channels, random_state, label, np.asarray(pobj),
            np.asarray(times), np.asarray(d_hat), np.asarray(z_hat), n_atoms,
            n_times_atom, n_trials, n_times, reg)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Programme to launch experiemnt')
    parser.add_argument('--njobs', type=int, default=1,
                        help='number of cores used to run the experiment')
    parser.add_argument('--dense', action="store_true",
                        help='run the experiment for multivariate')
    parser.add_argument('--wohlberg', action="store_true",
                        help='run the experiment for wohlberg')

    args = parser.parse_args()

    cached_one_run = mem.cache(func=one_run)

    all_results = []
    # load somato data
    from alphacsc.datasets.somato import load_data
    X, info = load_data(epoch=False, n_jobs=args.njobs)

    reg = .001
    n_iter = 50
    # number of random states
    n_states = 10
    n_channels = X.shape[1]
    span_channels = np.unique(np.floor(
        np.logspace(0, np.log10(n_channels), 10)).astype(int))
    methods = [
        [run_multichannel, 'rank1', n_iter],
        # [run_multivariate, 'dense', n_iter],
        # [run_multivariate, 'wohlberg', n_iter],
    ]

    if args.dense:
        methods = [[run_multivariate, 'dense', n_iter]]
        span_channels = np.unique(np.floor(
            np.logspace(0, np.log10(n_channels), 10)).astype(int))[:5]

    if args.wohlberg:
        methods = [[run_cbpdn, 'wohlberg', n_iter]]
        span_channels = np.unique(np.floor(
            np.logspace(0, np.log10(n_channels), 10)).astype(int))[:5]

    with Parallel(n_jobs=args.njobs) as parallel:

        iterator = itertools.product(range(n_states), methods, span_channels)
        # run the methods for different random_state
        delayed_one_run = delayed(cached_one_run)
        results = parallel(
            delayed_one_run(X, n_chan, method, n_atoms,
                            n_times_atom, rst, reg)
            for rst, method, n_chan in iterator)

        all_results.extend(results)

    # save even intermediate results
    suffix = ""
    if args.dense:
        suffix = "_dense"
    if args.wohlberg:
        suffix = "_wohlberg"

    all_results_df = pd.DataFrame(
        all_results, columns='n_channels random_state label pobj times '
        'd_hat z_hat n_atoms n_times_atom n_trials n_times reg'.split(' '))
    all_results_df.to_pickle(save_name.format(reg, suffix))
    import IPython
    IPython.embed()

    print('-- End of the script --')
