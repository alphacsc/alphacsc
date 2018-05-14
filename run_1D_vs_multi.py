import os
import itertools
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.externals.joblib import Parallel, delayed, Memory

from alphacsc.simulate import get_atoms
from alphacsc.update_d_multi import prox_uv
from alphacsc.utils import construct_X_multi
from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.utils.dictionary import get_D, get_lambda_max

verbose = 1
random_state = 27
rng = np.random.RandomState(random_state)


def find_best_allocation(value, order="max"):
    n_atoms = value.shape[0]
    matching = range(n_atoms)
    if order == 'max':
        best_ = 0
    else:
        best_ = 1e10
    for permutation in itertools.permutations(range(n_atoms)):
        current_ = abs(value[range(n_atoms), permutation]).sum()
        if order == 'max':
            if current_ > best_:
                matching = permutation
                best_ = current_
        else:
            if current_ < best_:
                matching = permutation
                best_ = current_
    return best_, matching


def compute_score(v, v_hat):
    return np.sum((v - v_hat)**2) / np.sum(v**2)


def score_D(uv, D_hat, n_channels):
    D = get_D(uv, n_channels)

    distances = np.array([[
        1 - abs(np.sum([np.correlate(Dkp_hat, Dkp, mode='valid').max()
                        for Dkp, Dkp_hat in zip(Dk, Dk_hat)])) / np.sum(Dk * Dk)
        for Dk_hat in D_hat] for Dk in D
    ])
    return find_best_allocation(distances, order='min')[0]


def score_uv(uv, uv_hat, n_channels):

    distances = np.array([[
        1 - abs(np.correlate(vk_hat, vk, mode='valid')).max() / np.sum(vk * vk)
        for vk_hat in uv_hat[:, n_channels:]] for vk in uv[:, n_channels:]
    ])
    return find_best_allocation(distances, order='min')[0]

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

    distances = np.array([[
        1 - abs(np.correlate(vk_hat, vk, mode='valid')).max() / np.sum(vk * vk)
        for vk_hat in d_hat] for vk in uv[:, n_channels:]
    ])
    return find_best_allocation(distances, order='min')[0]
    correlation = np.dot(d_hat, uv[:, n_channels:].T)
    max_corr, matching = find_best_allocation(correlation)

    v = uv[:, n_channels:]
    v_hat = d_hat[list(matching)]
    return compute_score(v, v_hat)


def get_signals(n_channels=50, n_times_atom=64, n_times_valid=640,
                sigma=.01, random_state=None):

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
    Z = np.array([sparse.random(n_atoms, n_times_valid, density=.05,
                                random_state=random_state).toarray()
                 for _ in range(n_trials)])
    Z = np.swapaxes(Z, 0, 1)

    X = construct_X_multi(Z, uv, n_channels=n_channels)

    X = X + sigma * rng.randn(*X.shape)

    uv_init = rng.randn(n_atoms, n_channels + n_times_atom)
    uv_init = prox_uv(uv_init, uv_constraint='separate', n_chan=n_channels)

    return X, uv, uv_init


def run_one(reg, sigma, n_atoms, n_times_atom, max_n_channels, n_times_valid,
            n_iter, run_n_channels, random_state):

    X, uv, uv_init = get_signals(max_n_channels, n_times_atom, n_times_valid,
                                 sigma, random_state)

    reg_ = reg * get_lambda_max(X, uv_init).max()
    reg_ *= run_n_channels

    uv_init_ = prox_uv(np.c_[uv_init[:, :run_n_channels],
                             uv_init[:, max_n_channels:]])
    uv_ = prox_uv(np.c_[uv[:, :run_n_channels], uv[:, max_n_channels:]],
                  uv_constraint='separate', n_chan=max_n_channels)

    def cb(X, uv_hat, Z_hat, pobj):
        it = len(pobj) // 2
        if it % 10 == 0:
            print("[channels{}] iter{} score sig={:.2e}: {:.3e}".format(
                run_n_channels, it, sigma,
                score_uv(uv_, uv_hat, run_n_channels)))

    pobj, times, uv_hat, Z_hat = learn_d_z_multi(
        X[:, :run_n_channels, :], n_atoms, n_times_atom,
        random_state=random_state,
        # callback=cb,
        n_iter=n_iter, n_jobs=1, reg=reg_, uv_constraint='separate',
        solver_d='alternate_adaptive', solver_d_kwargs={'max_iter': 50},
        solver_z='gcd', solver_z_kwargs=dict(tol=1e-3, maxiter=500),
        use_sparse_z=True, D_init=uv_init_, verbose=0,
    )

    score = score_uv(uv_, uv_hat, run_n_channels)
    print("=" * 79 + "\n"
          + "[channels{}-{}] iter{} score sig={:.2e}: {:.3e}\n"
          .format(run_n_channels, reg, len(pobj) // 2, sigma, score)
          + "=" * 79)

    return random_state, sigma, run_n_channels, score, uv, uv_hat, reg


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser('Programme to launch experiemnt')
    parser.add_argument('--njobs', type=int, default=6,
                        help='Number of processes used to run the experiement')
    args = parser.parse_args()

    save_name = 'rank1_snr.pkl'
    if not os.path.exists("figures"):
        os.mkdir("figures")
    save_name = os.path.join('figures', save_name)

    mem = Memory(cachedir='.', verbose=0)
    cached_run_one = mem.cache(func=run_one)
    delayed_run_one = delayed(cached_run_one)

    # Generate synchronous D
    n_times_atom = 64
    n_times_valid = 640
    max_n_channels = 50
    n_atoms = 2
    n_run = 1
    n_trials = 100
    strongest_ch = 0

    n_iter = 500

    span_reg = np.logspace(-5, -0.5, 15)[7:-2]
    span_random_state = np.arange(n_run)
    span_noise = np.logspace(-6, -1, 20)[5::2]
    span_channels = np.round(np.logspace(0, np.log10(max_n_channels), 10)
                             ).astype(int)
    spane_channels = [1, 2, 4, 9, 21, 50]

    grid_args = itertools.product(span_random_state, span_noise, span_channels,
                                  span_reg)

    results = Parallel(n_jobs=args.njobs)(
        delayed_run_one(reg_, sigma, n_atoms, n_times_atom, max_n_channels,
                        n_times_valid, n_iter, run_n_channels, random_state)
        for random_state, sigma, run_n_channels, reg_ in grid_args
    )

    # save even intermediate results
    all_results_df = pd.DataFrame(
        results,
        columns='random_state sigma run_n_channels score uv uv_hat reg'.split(' '))
    all_results_df.to_pickle(save_name)

    # plt.plot(uv_hat[:, n_channels:].T, 'g', label='Multivariate')
    # plt.plot(d_hat.T, 'r', label='1D')
    # plt.plot(uv[:, n_channels:].T, 'k--', label='ground truth')
    # plt.legend()
    # plt.savefig("figures/univariate_vs_multi.png", dpi=150)
    # plt.show()

    # score_uv(uv, uv_hat)
