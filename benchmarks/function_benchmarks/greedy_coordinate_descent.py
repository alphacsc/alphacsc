"""
0.5 . ||X - d * z ||_2^2 + lambda ||z||_1

"""
import numpy as np
import matplotlib.pyplot as plt
from joblib import delayed, Parallel

from alphacsc.utils.dictionary import get_lambda_max
from alphacsc.update_z_multi import _update_z_multi_idx


def _general_cd(X, D, reg, n_iter, strategy, n_seg):
    n_trials, n_channels, n_times = X.shape
    n_atoms, n_channels, n_times_atom = D.shape
    n_times_valid = n_times - n_times_atom + 1

    tol = 1e-7

    if n_seg == 'auto':
        if strategy == 'greedy':
            n_seg = max(n_times_valid // (2 * n_times_atom), 1)
        elif strategy in ('random', 'cyclic'):
            n_seg = 1
    max_iter = n_iter / float(n_seg) * n_times_valid * n_atoms

    solver_kwargs = dict(strategy=strategy, n_seg=n_seg, max_iter=max_iter,
                         tol=tol)

    z, pobj, times = _update_z_multi_idx(
        X, D, reg=reg, z0=None, idxs=np.arange(n_trials), debug=False,
        solver="lgcd", timing=True, solver_kwargs=solver_kwargs)
    return z, pobj, times


def gcd(X, D, reg, n_iter):
    strategy = 'greedy'
    n_seg = 1
    return _general_cd(X, D, reg, n_iter, strategy, n_seg)


def rcd(X, D, reg, n_iter):
    strategy = 'random'
    n_seg = 'auto'
    return _general_cd(X, D, reg, n_iter, strategy, n_seg)


def cd(X, D, reg, n_iter):
    strategy = 'cyclic'
    n_seg = 'auto'
    return _general_cd(X, D, reg, n_iter, strategy, n_seg)


def lgcd(X, D, reg, n_iter):
    strategy = 'greedy'
    n_seg = 'auto'
    return _general_cd(X, D, reg, n_iter, strategy, n_seg)


def _other_solver(X, D, reg, n_iter, solver, solver_kwargs):
    n_trials, n_channels, n_times = X.shape
    n_atoms, n_channels, n_times_atom = D.shape
    n_times_valid = n_times - n_times_atom + 1
    z0 = np.zeros((n_atoms, 1, n_times_valid))

    z, pobj, times = _update_z_multi_idx(
        X, D, reg, z0, idxs=np.arange(n_trials), debug=False, solver=solver,
        solver_kwargs=solver_kwargs, timing=True)

    return z, pobj, times


def l_bfgs(X, D, reg, n_iter):
    solver = 'l-bfgs'
    solver_kwargs = dict(factr=1e1, maxiter=n_iter - 1)

    z, pobj, times = _other_solver(X, D, reg, n_iter, solver, solver_kwargs)
    # Issue: The check in l_bfgs of parameter 'bounds'
    # (which change np.inf into None) adds an small overhead.

    return z, pobj, times


def ista(X, D, reg, n_iter):
    solver = 'ista'
    solver_kwargs = dict(power_iteration_tol=1e-4, max_iter=n_iter, eps=1e-10,
                         scipy_line_search=False)
    return _other_solver(X, D, reg, n_iter, solver, solver_kwargs)


def fista(X, D, reg, n_iter):
    solver = 'fista'
    solver_kwargs = dict(power_iteration_tol=1e-4, max_iter=n_iter, eps=1e-10,
                         scipy_line_search=False, restart=10)
    return _other_solver(X, D, reg, n_iter, solver, solver_kwargs)


# (func, max_iter)
all_func = [
    (cd, 20),
    (rcd, 20),
    (gcd, 10),
    (lgcd, 10),
    (l_bfgs, 200),
    (ista, 200),
    (fista, 200),
]


def run_one(func, n_times, n_atoms, n_times_atom, reg, n_iter, X, D):
    z, pobj, times = func(X, D, reg, n_iter)
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
        style = '-' if 'cd' in func.__name__ else '--'
        func_name, n_times, n_atoms, n_times_atom, reg, times, pobj = res
        plt.loglog(times, np.array(pobj) - best, style, label=func.__name__)
    plt.legend()
    plt.xlim(1e-2, None)
    name = ('T=%s_K=%s_L=%s_reg=%.3f' % (n_times, n_atoms, n_times_atom,
                                         reg_ratio))
    plt.title(name)
    plt.xlabel('Time (s)')
    plt.ylabel('loss function')
    save_name = 'figures/bench_gcd/' + name + '.png'
    print('Saving %s' % (save_name, ))
    fig.savefig(save_name)
    # plt.show()
    plt.close(fig)


def benchmark():
    pass


if __name__ == '__main__':
    reg_list = np.linspace(0.1, 0.9, 9)
    # reg_list = [0.2]
    n_jobs = 5
    Parallel(n_jobs=n_jobs)(delayed(plot_loss)(reg) for reg in reg_list)

    benchmark()
