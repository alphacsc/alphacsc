
import IPython
import itertools

from sklearn.externals.joblib import Memory, Parallel, delayed


from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.utils.viz import get_callback_csc
from alphacsc.datasets.somato import load_data


mem = Memory(cachedir='.', verbose=0)


def run_one(X, csc_kwargs, topo):
    return learn_d_z_multi(
        X, n_jobs=1,
        callback=get_callback_csc(csc_kwargs, sfreq, plot_topo=topo),
        **csc_kwargs)


_run_one_cached = mem.cache(run_one)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Programme to launch experiment')
    parser.add_argument('--debug', action='store_true',
                        help='Debug run with the njobs first term in the grid')
    parser.add_argument('--topo', action='store_true',
                        help='Debug run with the njobs first term in the grid')
    parser.add_argument('--njobs', type=int, default=4,
                        help='Number of CPU one for multiprocessing exp.')
    args = parser.parse_args()

    sfreq = 300

    X, plot_topo = load_data(sfreq=sfreq)
    if not args.topo:
        plot_topo = None

    default_kwargs = dict(
        n_atoms=25, n_times_atom=int(sfreq * .2),
        random_state=42, n_iter=50, reg=.5, eps=1e-3,
        solver_z="gcd", solver_z_kwargs={'max_iter': 1000},
        solver_d='alternate_adaptive', solver_d_kwargs={'max_iter': 100},
        uv_init='ssa', uv_constraint='separate', loss='l2', gamma=.005,
        algorithm='batch', lmbd_max='shared'
    )

    grid = dict(
        n_atoms=[10, 20, 30],
        n_times_atom=[int(sfreq * r) for r in [.05, .1, .15, .2, .3]],
        reg=[.7, .5, .4, .3, .2, .1]
    )
    kwargs_grid = [{**default_kwargs, **dict(zip(grid, val))}
                   for val in itertools.product(*grid.values())]

    if args.debug:
        kwargs_grid = kwargs_grid[:args.njobs]

    with Parallel(n_jobs=args.njobs) as parallel:
        delayed_run_one = delayed(_run_one_cached)
        res = parallel(delayed_run_one(X, csc_kwargs, plot_topo)
                       for csc_kwargs in kwargs_grid)

    IPython.embed()
