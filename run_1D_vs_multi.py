import os
import itertools
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.externals.joblib import Parallel, delayed, Memory

from alphacsc.simulate import get_atoms
from alphacsc.learn_d_z import learn_d_z
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
                        for Dkp, Dkp_hat in zip(Dk, Dk_hat)]) / np.sum(Dk * Dk)
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


def run_one(X, D_init, sigma, uv, n_atoms, n_times_atom, n_channels, n_iter,
            strongest_ch):

    X = X + sigma * rng.randn(*X.shape)
    X_uni = X[:, strongest_ch]
    D_init_uni = D_init[:, n_channels:]
    D_init_dense = get_D(D_init, n_channels)

    reg_multi = reg * get_lambda_max(X, D_init).max()
    reg_uni = reg * get_lambda_max(X_uni[:, None], D_init_uni[:, None]).max()
    reg_dense= reg * get_lambda_max(X_uni, D_init_dense).max()

    # print('Univariate gcd:')
    # uv_init = np.c_[np.ones((n_atoms, 1)), D_init[:, n_channels:]]
    # pobj, times, uv_hat_uni, Z_hat = learn_d_z_multi(
    #     X[:, strongest_ch:strongest_ch + 1, :], n_atoms, n_times_atom,
    #     random_state=random_state, callback=None,
    #     n_iter=n_iter, n_jobs=1, reg=reg / 5, uv_constraint='separate',
    #     solver_z='gcd', solver_z_kwargs=dict(tol=1e-3, maxiter=200),
    #     solver_d='alternate_adaptive', solver_d_kwargs={'max_iter': 100},
    #     use_sparse_z=True, D_init=uv_init, verbose=1,
    # )
    # uv_hat_uni = np.c_[
    #     np.ones((n_atoms, 1)),
    #     uv_hat_uni[:, 1:] * uv_hat_uni[:, :1]]

    # print("done\nUnivariate_gcd score sig={:.2e}: {:.3e}".format(
    #       sigma, score_d(uv, uv_hat_uni[:, 1:], n_channels)))
    uv_hat_uni = np.c_[np.ones((n_atoms, 1)), D_init[:, n_channels:]]

    # print('Univariate:')
    # pobj, times, d_hat, Z_hat = learn_d_z(
    #     X_uni, n_atoms, n_times_atom,
    #     reg=reg_uni, n_iter=n_iter * 5, ds_init=D_init_uni,
    #     solver_d_kwargs=dict(factr=100, max_iter=100),
    #     random_state=random_state, n_jobs=1, verbose=verbose)

    # print("done\nUnivariate score sig={:.2e}: {:.3e}".format(
    #       sigma, score_d(uv, d_hat, n_channels)))

    # print('Multivariate:')
    # # from alphacsc.utils.viz import get_callback_csc
    # csc_kwargs = dict(
    #     n_atoms=n_atoms, n_times_atom=n_times_atom, random_state=random_state,
    #     n_iter=n_iter, n_jobs=1, reg=reg_multi,
    #     uv_constraint='separate',
    #     solver_z='gcd', solver_z_kwargs=dict(tol=1e-3, maxiter=200),
    #     solver_d='alternate_adaptive', solver_d_kwargs={'max_iter': 200}
    # )
    # pobj, times, uv_hat, Z_hat = learn_d_z_multi(
    #     X, **csc_kwargs,
    #     use_sparse_z=True, D_init=D_init, verbose=verbose,
    #     callback=None
    #     # callback=get_callback_csc(csc_kwargs, config={'atom': {}})
    # )
    # print("done\nMultivariate score sig={:.2e}: {:.3e}".format(
    #       sigma, score_uv(uv, uv_hat, n_channels)))

    print('Dense:')
    from alphacsc.utils.viz import get_callback_csc
    csc_kwargs = dict(
        n_atoms=n_atoms, n_times_atom=n_times_atom, random_state=random_state,
        n_iter=n_iter, n_jobs=1, reg=reg_dense,
        uv_constraint='separate', rank1=False,
        solver_z='gcd', solver_z_kwargs=dict(tol=1e-3, maxiter=200),
        solver_d='alternate_adaptive', solver_d_kwargs={'max_iter': 200}
    )
    pobj, times, uv_hat, Z_hat = learn_d_z_multi(
        X, **csc_kwargs,
        use_sparse_z=True, D_init=D_init_dense, verbose=verbose,
        # callback=None
        callback=get_callback_csc(csc_kwargs, config={'atom': {}})
    )
    print("done\nMultivariate score sig={:.2e}: {:.3e}".format(
          sigma, score_uv(uv, uv_hat, n_channels)))

    return (sigma, score_d(uv, d_hat, n_channels),
            score_d(uv, uv_hat_uni[:, 1:], n_channels),
            score_uv(uv, uv_hat, n_channels),
            uv, uv_hat, d_hat)


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
    n_times_atom, n_times = 64, 512
    n_times_atom_gen = 64
    n_times_valid = n_times - n_times_atom + 1
    n_channels = 50
    n_atoms = 2
    n_trials = 100
    strongest_ch = 0

    n_iter = 100

    v0 = get_atoms('triangle', n_times_atom_gen)  # temporal atoms
    v1 = get_atoms('square', n_times_atom_gen)

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

    reg = 0.2

    D_init = rng.randn(n_atoms, n_channels + n_times_atom)
    D_init = prox_uv(D_init, uv_constraint='separate', n_chan=n_channels)

    span_sigma = np.logspace(-6, -1, 20)
    results = Parallel(n_jobs=args.njobs)(
        delayed_run_one(X, D_init, sigma, uv, n_atoms, n_times_atom,
                        n_channels, n_iter, strongest_ch)
        for sigma in span_sigma
    )

    # save even intermediate results
    all_results_df = pd.DataFrame(
        results, columns='sigma score_d score_uv_uni score_uv uv uv_hat '
        'd_hat'.split(' '))
    all_results_df.to_pickle(save_name)

    # plt.plot(uv_hat[:, n_channels:].T, 'g', label='Multivariate')
    # plt.plot(d_hat.T, 'r', label='1D')
    # plt.plot(uv[:, n_channels:].T, 'k--', label='ground truth')
    # plt.legend()
    # plt.savefig("figures/univariate_vs_multi.png", dpi=150)
    # plt.show()

    # score_uv(uv, uv_hat)
