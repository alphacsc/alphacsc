from __future__ import print_function
import os
import itertools
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.externals.joblib import Parallel, delayed, Memory

from alphacsc.utils import check_random_state
from alphacsc.simulate import simulate_data
from alphacsc.update_d import update_d_block
from alphacsc.learn_d_z import learn_d_z
from alphacsc.learn_d_z_multi import learn_d_z_multi

mem = Memory(cachedir='.', verbose=0)

START = time.time()

##############################
# Parameters of the simulation
verbose = 1

# n_jobs for the parallel running of single core methods
n_jobs = 5
# number of random states
n_states = 1

n_trials = 10  # N
n_times_atom = 16  # L
n_times = 2000  # T
n_atoms = 2  # K
reg = 1.0

# A method stops if its objective function reaches best_pobj * (1 + threshold)
threshold = -1

save_name = 'methods_th%s_' % (threshold, )
save_name = os.path.join('figures', save_name)


###################
# simulate the data
@mem.cache()
def simulation(n_atoms, n_times_atom, n_trials, n_times, reg):
    n_atoms = min(n_atoms, 12)  # maximum n_atoms in current simulate_data
    random_state_simulate = 10
    X, ds_true, Z_true = simulate_data(
        n_trials=n_trials, n_times=n_times, n_times_atom=n_times_atom,
        n_atoms=n_atoms, random_state=random_state_simulate)

    # scaling to have a scaling of reg that is reasonable
    X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(
        X, axis=1, keepdims=True)

    rng = check_random_state(random_state_simulate)
    X += 0.1 * rng.randn(*X.shape)

    # set borders of X to 0.
    pad = np.zeros((n_trials, n_times_atom * 1))
    X = np.hstack([pad, X, pad])

    ###############
    # best possible
    print('Computing best possible obj')
    ds_true = np.pad(ds_true, [(0, 0), (n_times_atom // 2,
                                        n_times_atom // 2 + 1)],
                     mode='constant')
    n_atoms, n_times_atom = ds_true.shape
    pobj, _, _, _ = learn_d_z(X, n_atoms, n_times_atom, func_d=update_d_block,
                              solver_z='l_bfgs', solver_z_kwargs=dict(
                                  factr=1e11), reg=reg, n_iter=100,
                              random_state=random_state_simulate,
                              ds_init=ds_true, n_jobs=n_jobs, verbose=1)
    best_pobj = pobj[-1]
    print('[Done]')
    return X, best_pobj


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
              factr_d=1e7, factr_z=1e15):
    n_atoms, n_times_atom = ds_init.shape
    pobj, times, d_hat, z_hat = learn_d_z(
        X, n_atoms, n_times_atom,
        func_d=update_d_block, solver_z='l_bfgs', solver_z_kwargs=dict(
            factr=factr_z), reg=reg, n_iter=n_iter, solver_d_kwargs=dict(
                factr=factr_d), random_state=random_state, ds_init=ds_init,
        n_jobs=1, stopping_pobj=stopping_pobj, verbose=verbose)

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


def run_multichannel_joint(X, ds_init, reg, n_iter, random_state, label,
                           stopping_pobj):
    n_atoms, n_times_atom = ds_init.shape
    uv_init = np.c_[np.ones((n_atoms, 1)), ds_init]
    pobj, times, d_hat, z_hat = learn_d_z_multi(
        X[:, None, :], n_atoms, n_times_atom,
        solver_d='joint', uv_constraint='separate', solver_z_kwargs=dict(
            factr=1e15), reg=reg, solver_d_kwargs=dict(max_iter=10),
        n_iter=n_iter, random_state=random_state, uv_init=uv_init, n_jobs=1,
        stopping_pobj=stopping_pobj, verbose=verbose)

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


def run_multichannel_separate(X, ds_init, reg, n_iter, random_state, label,
                              stopping_pobj):
    n_atoms, n_times_atom = ds_init.shape
    uv_init = np.c_[np.ones((n_atoms, 1)), ds_init]
    pobj, times, d_hat, z_hat = learn_d_z_multi(
        X[:, None, :], n_atoms, n_times_atom,
        solver_d='alternate', uv_constraint='separate', solver_z_kwargs=dict(
            factr=1e15), reg=reg, solver_d_kwargs=dict(max_iter=10),
        n_iter=n_iter, random_state=random_state, uv_init=uv_init, n_jobs=1,
        stopping_pobj=stopping_pobj, verbose=verbose)

    return pobj[::2], np.cumsum(times)[::2], d_hat, z_hat


methods = [
    [run_ista, 'vanilla_ista', 500],
    [run_fista, 'vanilla_fista', 500],
    [run_lbfgs, 'vanilla_lbfgsb ', 500],
    [run_multichannel_joint, 'multi_joint_s', 500],
    [run_multichannel_separate, 'multi_alternate_se', 500],
]


def one_run(X, X_shape, random_state, method, n_atoms, n_times_atom,
            stopping_pobj, best_pobj):
    n_trials, n_times = X_shape
    func, label, n_iter = method
    current_time = time.time() - START
    print('%s - %s: started at %.0f sec' % (random_state, label, current_time))

    # use the same init for all methods
    rng = check_random_state(random_state)
    ds_init = rng.randn(n_atoms, n_times_atom * 2 + 1)

    # run the selected algorithm with one iter to remove compilation overhead
    _, _, _, _ = func(X, ds_init, reg, 1, random_state, label, stopping_pobj)

    # run the selected algorithm
    pobj, times, d_hat, z_hat = func(X, ds_init, reg, n_iter, random_state,
                                     label, stopping_pobj)

    # store z_hat in a sparse matrix to reduce size
    for z in z_hat:
        z[z < 1e-3] = 0
    z_hat = [sp.csr_matrix(z) for z in z_hat]

    current_time = time.time() - START
    print('%s - %s: done at %.0f sec' % (random_state, label, current_time),
          end='')
    return (random_state, label, np.asarray(pobj), np.asarray(times),
            np.asarray(d_hat), np.asarray(z_hat), n_atoms, n_times_atom,
            n_trials, n_times, stopping_pobj, best_pobj)


if __name__ == '__main__':

    cached_one_run = mem.cache(func=one_run, ignore=['X'])

    all_results = []
    for n_atoms in [
            2,
            # 10,
    ]:
        for n_times_atom in [
                32,
                # 128,
        ]:
            print(n_atoms, n_times_atom)
            # simulate the data and optimize to get the best pobj
            X, best_pobj = simulation(n_atoms, n_times_atom, n_trials, n_times,
                                      reg)
            X_shape = X.shape
            stopping_pobj = best_pobj * (1 + threshold)

            # run the methods for different random_state
            delayed_one_run = delayed(cached_one_run)
            results = Parallel(n_jobs=n_jobs)(
                delayed_one_run(X, X_shape, random_state, method, n_atoms,
                                n_times_atom, stopping_pobj, best_pobj)
                for method, random_state in itertools.product(
                    methods, range(n_states)))

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
