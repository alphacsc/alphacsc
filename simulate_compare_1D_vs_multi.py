import os
import itertools
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.externals.joblib import Parallel, delayed, Memory

from alphacsc.simulate import get_atoms
from alphacsc.utils import construct_X_multi
from alphacsc.update_d_multi import prox_uv
from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.learn_d_z import learn_d_z


random_state = 27
rng = np.random.RandomState(random_state)


def find_best_allocation(correlation):
    n_atoms = correlation.shape[0]
    matching = range(n_atoms)
    max_corr = 0
    for permutation in itertools.permutations(range(n_atoms)):
        curr_corr = abs(correlation[range(n_atoms), permutation]).sum()
        if curr_corr > max_corr:
            matching = permutation
            max_corr = curr_corr
    return max_corr, matching


def compute_score(v, v_hat):
    return np.sum((v - v_hat)**2) / np.sum(v**2)


def score_uv(uv, uv_hat, n_channels):
    n_atoms = uv.shape[0]

    correlation = np.dot(uv_hat[:, :n_channels], uv[:, :n_channels].T)
    max_corr, matching = find_best_allocation(correlation)
    uv_hat = uv_hat[list(matching)]

    # Find the right orientation:
    for k in range(n_atoms):
        if np.dot(uv[k, :n_channels], uv_hat[k, :n_channels]) < 0:
            uv_hat[k] *= -1

    v = uv[:, n_channels:]
    v_hat = uv_hat[:, n_channels:]
    return compute_score(v, v_hat)


def score_d(uv, d_hat, n_channels):

    correlation = np.dot(d_hat, uv[:, n_channels:].T)
    max_corr, matching = find_best_allocation(correlation)

    v = uv[:, n_channels:]
    v_hat = d_hat[list(matching)]
    return compute_score(v, v_hat)


def run_one(X, D_init, sigma, uv, n_atoms, n_times_atom, n_channels, n_iter,
            strongest_ch):

    X = X + sigma * rng.randn(*X.shape)

    print('Univariate:')
    pobj, times, d_hat, Z_hat = learn_d_z(
        X[:, strongest_ch, :], n_atoms, n_times_atom,
        reg=reg, n_iter=n_iter, ds_init=D_init[:, n_channels:],
        solver_d_kwargs=dict(factr=100), random_state=random_state,
        n_jobs=1, verbose=1)

    print('done\nMultivariate:')
    pobj, times, uv_hat, Z_hat = learn_d_z_multi(
        X, n_atoms, n_times_atom, random_state=random_state, callback=None,
        n_iter=n_iter, n_jobs=1, reg=reg, uv_constraint='separate',
        solver_z='gcd', solver_z_kwargs=dict(tol=1e-3, maxiter=200),
        solver_d='alternate_adaptive', solver_d_kwargs={'max_iter': 100},
        use_sparse_z=True, D_init=D_init, verbose=1,
    )

    return (sigma, score_d(uv, d_hat, n_channels),
            score_uv(uv, uv_hat, n_channels),
            uv, uv_hat, d_hat)

if __name__ == "__main__":

    save_name = 'rank1_snr.pkl'
    if not os.path.exists("figures"):
        os.mkdir("figures")
    save_name = os.path.join('figures', save_name)

    mem = Memory(cachedir='.', verbose=0)
    cached_run_one = mem.cache(func=run_one)
    delayed_run_one = delayed(cached_run_one)

    # Generate synchronous D
    n_times_atom, n_times = 64, 512
    n_times_valid = n_times - n_times_atom + 1
    n_channels = 50
    n_atoms = 2
    n_trials = 100
    strongest_ch = 0

    n_iter = 100

    v0 = get_atoms('triangle', n_times_atom)  # temporal atoms
    v1 = get_atoms('square', n_times_atom)

    u0 = get_atoms('sin', n_channels)  # spatial maps
    u1 = get_atoms('cos', n_channels)
    u0[strongest_ch] = u1[strongest_ch] = 1

    uv = np.array([np.r_[u0, v0], np.r_[u1, v1]])
    uv = prox_uv(uv, 'separate', n_channels)

    # add atoms
    shape_Z = (n_atoms, n_trials, n_times_atom)
    Z = np.array([sparse.random(n_atoms, n_times_valid, density=.05,
                                random_state=random_state).toarray()
                 for _ in range(n_trials)])
    Z = np.swapaxes(Z, 0, 1)
    assert np.all(Z >= 0)

    X = construct_X_multi(Z, uv, n_channels=n_channels)

    reg = 0.01

    D_init = rng.randn(n_atoms, n_channels + n_times_atom)
    D_init = prox_uv(D_init, uv_constraint='separate', n_chan=n_channels)

    span_sigma = np.logspace(-3, -1, 12)
    results = Parallel(n_jobs=4)(
        delayed_run_one(X, D_init, sigma, uv, n_atoms, n_times_atom,
                        n_channels, n_iter, strongest_ch)
        for sigma in span_sigma
    )

    # save even intermediate results
    all_results_df = pd.DataFrame(
        results, columns='sigma score_d score_uv uv uv_hat d_hat'.split(' '))
    all_results_df.to_pickle(save_name)

    # plt.plot(uv_hat[:, n_channels:].T, 'g', label='Multivariate')
    # plt.plot(d_hat.T, 'r', label='1D')
    # plt.plot(uv[:, n_channels:].T, 'k--', label='ground truth')
    # plt.legend()
    # plt.savefig("figures/univariate_vs_multi.png", dpi=150)
    # plt.show()

    # score_uv(uv, uv_hat)
