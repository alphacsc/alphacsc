"""
0.5 . ||X - d * z ||_2^2 + lambda ||z||_1

"""
import numpy as np
import matplotlib.pyplot as plt

from alphacsc.update_z_multi import _update_z_multi_idx
from alphacsc.utils.dictionary import get_lambda_max


def _general_cd(X, D, reg, n_iter, strategy, n_seg):
    n_trials, n_channels, n_times = X.shape
    solver_kwargs = dict(strategy=strategy, n_seg=n_seg,
                         max_iter=n_iter / float(n_seg), tol=1e-4)
    Z, pobj, times = _update_z_multi_idx(
        X, D, reg=reg, z0=None, idxs=np.arange(n_trials), debug=False,
        solver='gcd', timing=True, solver_kwargs=solver_kwargs)
    return Z, pobj, times


def gcd(X, D, reg, n_iter):
    strategy = 'greedy'
    n_seg = 1
    return _general_cd(X, D, reg, n_iter, strategy, n_seg)


def cd(X, D, reg, n_iter):
    strategy = 'greedy'
    n_trials, n_channels, n_times = X.shape
    n_atoms, n_channels, n_times_atom = D.shape
    n_times_valid = n_times - n_times_atom + 1
    n_seg = n_times_valid
    return _general_cd(X, D, reg, n_iter, strategy, n_seg)


def lgcd(X, D, reg, n_iter):
    strategy = 'greedy'
    n_trials, n_channels, n_times = X.shape
    n_atoms, n_channels, n_times_atom = D.shape
    n_times_valid = n_times - n_times_atom + 1
    n_seg = max(n_times_valid // (2 * n_times_atom), 1)
    return _general_cd(X, D, reg, n_iter, strategy, n_seg)


def _other_solver(X, D, reg, n_iter, solver, solver_kwargs):
    n_trials, n_channels, n_times = X.shape
    n_atoms, n_channels, n_times_atom = D.shape
    n_times_valid = n_times - n_times_atom + 1
    Z0 = np.zeros((n_atoms, 1, n_times_valid))

    Z, pobj, times = _update_z_multi_idx(
        X, D, reg, Z0, idxs=np.arange(n_trials), debug=False, solver=solver,
        solver_kwargs=solver_kwargs, timing=True)

    return Z, pobj, times


def lbfgs(X, D, reg, n_iter):
    solver = 'l_bfgs'
    solver_kwargs = dict(factr=1e1, maxiter=n_iter - 1)

    Z, pobj, times = _other_solver(X, D, reg, n_iter, solver, solver_kwargs)
    # Issue: The check in lbfgs of parameter 'bounds'
    # (which change np.inf into None) adds an small overhead.

    return Z, pobj, times


def ista(X, D, reg, n_iter):
    solver = 'ista'
    solver_kwargs = dict(power_iteration_tol=1e-4, max_iter=n_iter)
    return _other_solver(X, D, reg, n_iter, solver, solver_kwargs)


def fista(X, D, reg, n_iter):
    solver = 'fista'
    solver_kwargs = dict(power_iteration_tol=1e-4, max_iter=n_iter,
                         restart=None)
    return _other_solver(X, D, reg, n_iter, solver, solver_kwargs)


# (func, max_iter)
all_func = [
    (cd, 100000),
    (gcd, 100000),
    (lgcd, 100000),
    (lbfgs, 200),
    (ista, 100),
    (fista, 100),
]


def run_one(func, n_times, n_atoms, n_times_atom, reg, n_iter, X, D):
    Z, pobj, times = func(X, D, reg, n_iter)
    times = np.cumsum(times)

    return (func.__name__, n_times, n_atoms, n_times_atom, reg, times, pobj)


def plot_loss(reg_ratio):
    n_trials = 1
    n_channels = 1
    n_times, n_atoms, n_times_atom = 100000, 10, 100

    rng = np.random.RandomState(0)
    X = rng.randn(n_trials, n_channels, n_times)
    D = rng.randn(n_atoms, n_channels, n_times_atom)

    reg = reg_ratio * get_lambda_max(X, D).max()

    results = []
    for func, max_iter in all_func:
        print(func.__name__)
        res = run_one(func, n_times, n_atoms, n_times_atom, reg, max_iter, X,
                      D)
        results.append(res)

    best = np.inf
    for res in results:
        func_name, n_times, n_atoms, n_times_atom, reg, times, pobj = res
        if pobj[-1] < best:
            best = pobj[-1]

    fig = plt.figure()
    for (func, max_iter), res in zip(all_func, results):
        style = '-' if 'cd' in func.__name__ else '.--'
        func_name, n_times, n_atoms, n_times_atom, reg, times, pobj = res
        plt.semilogy(times, pobj - best, style, label=func.__name__)
    plt.legend()
    name = 'reg=%.3e_T=_K=_L=' % (reg_ratio, n_times, n_atoms, n_times_atom)
    plt.title(name)
    plt.xlabel('Time (s)')
    plt.ylabel('loss function')
    fig.savefig('figures/bench_gcd/' + name + '.png')
    plt.close(fig)


def benchmark():
    pass


if __name__ == '__main__':
    reg_list = np.logspace(np.log10(0.01), np.log10(0.9), 10)
    for reg_ratio in reg_list:
        plot_loss(reg_ratio)
    plt.show()

    benchmark()
