"""Benchmark multiple channels vs a single channel for dictionary recovery.

This script requires `pandas` which can be installed with `pip install pandas`.

This script performs the computations and save the results in a pickled file
`figures/rank1_snr.pkl` which can be plotted using `1D_vs_multi_plot.py`.
"""
import os
import itertools
import numpy as np
import pandas as pd
from scipy import sparse
from joblib import Parallel, delayed, Memory

from alphacsc.simulate import get_atoms
from alphacsc.update_d_multi import prox_uv
from alphacsc.utils import construct_X_multi
from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.utils.dictionary import get_lambda_max


VERBOSE = 0


############################################
# Scoring functions for dictionary recovery
############################################

def find_best_allocation(cost, order="min"):
    """Computes best matching between entries given a cost matrix and return
    cost."""
    n_atoms = cost.shape[0]
    if order == 'max':
        best_ = 0
    else:
        best_ = 1e10
    for permutation in itertools.permutations(range(n_atoms)):
        current_ = abs(cost[range(n_atoms), permutation]).sum()
        if order == 'max':
            if current_ > best_:
                best_ = current_
        else:
            if current_ < best_:
                best_ = current_
    return best_


def score_uv(uv, uv_hat, n_channels):
    """Compute the recovery score for uv_hat compared to uv."""

    distances = np.array([[
        1 - abs(np.correlate(vk_hat, vk, mode='valid')).max() / np.sum(vk * vk)
        for vk_hat in uv_hat[:, n_channels:]] for vk in uv[:, n_channels:]
    ])
    return find_best_allocation(distances, order='min')


#############################
# Signal generation function
#############################

def get_signals(n_channels=50, n_times_atom=64, n_times_valid=640,
                sigma=.01, random_state=None):
    """Generate a signal following the sparse linear model with a rank1
    triangle and square atoms and a Bernoulli-uniform distribution."""

    n_atoms = 2
    rng = np.random.RandomState(random_state)

    v0 = get_atoms('triangle', n_times_atom)  # temporal atoms
    v1 = get_atoms('square', n_times_atom)

    u0 = get_atoms('sin', n_channels)  # spatial maps
    u1 = get_atoms('cos', n_channels)
    u0[0] = u1[0] = 1

    uv = np.array([np.r_[u0, v0], np.r_[u1, v1]])
    uv = prox_uv(uv, 'separate', n_channels)

    # add atoms
    z = np.array([sparse.random(n_atoms, n_times_valid, density=.05,
                                random_state=random_state).toarray()
                 for _ in range(n_trials)])
    z = np.swapaxes(z, 0, 1)

    X = construct_X_multi(z, uv, n_channels=n_channels)

    X = X + sigma * rng.randn(*X.shape)

    uv_init = rng.randn(n_atoms, n_channels + n_times_atom)
    uv_init = prox_uv(uv_init, uv_constraint='separate', n_channels=n_channels)

    return X, uv, uv_init


####################################
# Calling function of the benchmark
####################################

def run_one(reg, sigma, n_atoms, n_times_atom, max_n_channels, n_times_valid,
            n_iter, run_n_channels, random_state):
    """Run the benchmark for a given set of parameter."""

    X, uv, uv_init = get_signals(max_n_channels, n_times_atom, n_times_valid,
                                 sigma, random_state)

    reg_ = reg * get_lambda_max(X, uv_init).max()
    # reg_ *= run_n_channels

    uv_init_ = prox_uv(np.c_[uv_init[:, :run_n_channels],
                             uv_init[:, max_n_channels:]])
    uv_ = prox_uv(np.c_[uv[:, :run_n_channels], uv[:, max_n_channels:]],
                  uv_constraint='separate', n_channels=max_n_channels)

    def cb(z_encoder, pobj):
        it = len(pobj) // 2
        if it % 10 == 0:
            uv_hat = z_encoder.D_hat
            print("[channels{}] iter{} score sig={:.2e}: {:.3e}".format(
                run_n_channels, it, sigma,
                score_uv(uv_, uv_hat, run_n_channels)))

    pobj, times, uv_hat, z_hat, reg = learn_d_z_multi(
        X[:, :run_n_channels, :], n_atoms, n_times_atom,
        random_state=random_state, rank1=True,
        # callback=cb,
        n_iter=n_iter, n_jobs=1, reg=reg_, uv_constraint='separate',
        solver_d='alternate_adaptive', solver_d_kwargs={'max_iter': 50},
        solver_z="lgcd", solver_z_kwargs=dict(tol=1e-3, maxiter=500),
        D_init=uv_init_, verbose=VERBOSE,
    )

    score = score_uv(uv_, uv_hat, run_n_channels)
    print("=" * 79 + "\n"
          + "[channels{}-{:.2e}-{}] iter {} score sig={:.2e}: {:.3e}\n"
          .format(run_n_channels, reg, random_state, len(pobj) // 2, sigma,
                  score) + "=" * 79)

    return random_state, sigma, run_n_channels, score, uv, uv_hat, reg


###############################
# Main script of the benchmark
###############################

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser('Benchmark to highlight the advantages '
                                     'of using multiple channels versus a one '
                                     'channel.')
    parser.add_argument('--njobs', type=int, default=6,
                        help='Number of processes used to run the benchmark.')
    args = parser.parse_args()

    # Use the caching utilities from joblib to same intermediate results and
    # avoid loosing computations when the interpreter crashes.
    mem = Memory(location='.', verbose=VERBOSE)
    cached_run_one = mem.cache(func=run_one)
    delayed_run_one = delayed(cached_run_one)

    # Generate synchronous D
    n_times_atom = 64
    n_times_valid = 640
    max_n_channels = 50
    n_atoms = 2
    n_run = 3
    n_trials = 100
    strongest_ch = 0

    n_iter = 500

    # Create a grid a parameter for which we which to run the benchmark.
    span_reg = np.logspace(-4, -0.5, 10)
    span_random_state = np.arange(n_run)
    span_noise = np.logspace(-4, -1, 10)
    span_channels = np.unique(
        np.round(np.logspace(0, np.log10(max_n_channels), 10)).astype(int))
    grid_args = itertools.product(span_random_state, span_noise, span_channels,
                                  span_reg)

    # Run the experiment in parallel with joblib
    results = Parallel(n_jobs=args.njobs)(
        delayed_run_one(reg_, sigma, n_atoms, n_times_atom, max_n_channels,
                        n_times_valid, n_iter, run_n_channels, random_state)
        for random_state, sigma, run_n_channels, reg_ in grid_args
    )

    # save all results for plotting with 1D_vs_multi_plot.py script.
    save_name = 'rank1_snr.pkl'
    if not os.path.exists("figures"):
        os.mkdir("figures")
    save_name = os.path.join('figures', save_name)
    all_results_df = pd.DataFrame(
        results, columns='random_state sigma run_n_channels '
                         'score uv uv_hat reg'.split(' '))
    all_results_df.to_pickle(save_name)
