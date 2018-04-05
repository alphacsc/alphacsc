import os
import shutil
import IPython

import argparse
import itertools
import importlib
from datetime import datetime
import matplotlib.pyplot as plt
from joblib import Memory, Parallel, delayed

from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.utils.viz import get_callback_csc, DEFAULT_CB
from alphacsc.utils.plot_output import DEFAULT_OUTPUT, PLOTS
from alphacsc.datasets import DATASETS


mem = Memory(cachedir='.', verbose=0)


def run_one(X, csc_kwargs, n_jobs=1, info={}, callback_config=DEFAULT_CB,
            verbose=0, run=0):
    name = "Run{}".format(run)
    print('------------- running %s %.2f----------------' % (
        name, csc_kwargs['reg']))
    callback = get_callback_csc(csc_kwargs, info=info, config=callback_config)
    n_iter = csc_kwargs.pop('n_iter', 50)
    _, _, D_init, Z_init = learn_d_z_multi(X, n_jobs=n_jobs, n_iter=0,
                                           name=name, **csc_kwargs)
    pobj, times, D_hat, Z_hat = learn_d_z_multi(
        X, n_jobs=n_jobs, n_iter=n_iter, name=name,
        verbose=verbose, callback=callback, **csc_kwargs)
    return dict(
        pobj=pobj, times=times, D_hat=D_hat, Z_hat=Z_hat, D_init=D_init,
        Z_init=Z_init
    )


run_one_cached = mem.cache(run_one, ignore=['callback_config', 'run',
                                            'info', 'n_jobs', 'verbose'])
run_one_delayed = delayed(run_one_cached)


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Main script for multi csc experiments.')
    parser.add_argument('--n_iter', type=int, default=50,
                        help='Print profiling of the function')
    parser.add_argument('--exp', type=str, default='simu',
                        help='Dataset used in the experiment.')
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='Dataset used in the experiment.')
    parser.add_argument('--debug', action="store_true",
                        help='Debug mode, clean-up the saved exp at the end '
                        'of the script.')

    args = parser.parse_args()

    # Load the experiment
    exp = importlib.import_module('exps.{}'.format(args.exp), package='exps')

    dirname = exp.__file__.replace('.py', '')
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    csc_time = datetime.now().strftime('%y_%m_%d_%Hh%M')
    dirname = os.path.join(dirname, csc_time)
    os.mkdir(dirname)

    shutil.copyfile('exps/{}.py'.format(args.exp),
                    '{}/{}.py'.format(dirname, args.exp))

    sfreq = getattr(exp, 'sfreq', 1)
    callback_config = getattr(exp, 'callback_config', {})
    output_config = getattr(exp, 'output_config', DEFAULT_OUTPUT)

    if exp.dataset == 'simu':
        from alphacsc.datasets.simulate import load_data
        dataset_kwargs = getattr(exp, 'dataset_kwargs', {})
        X, info = load_data(**dataset_kwargs)

    elif exp.dataset == 'somato':

        from alphacsc.datasets.somato import load_data
        X, info = load_data(sfreq=sfreq, epoch=False, n_jobs=args.n_jobs)

    else:
        raise NotImplementedError("No dataset named {}. Should select one in "
                                  "{}".format(exp.dataset, DATASETS))

    info['sfreq'] = sfreq
    info['n_channels'] = X.shape[1]

    try:
        default_kwargs = exp.default_kwargs
    except AttributeError:
        raise ValueError("default_kwargs should be defined in {}"
                         .format(args.exp))

    if args.debug:
        default_kwargs['n_iter'] = 1

    if hasattr(exp, 'grid'):
        kwargs_grid = [{**default_kwargs, **dict(zip(exp.grid, val))}
                       for val in itertools.product(*exp.grid.values())]

        with Parallel(n_jobs=args.n_jobs) as parallel:
            res = parallel(run_one_delayed(X, csc_kwargs, info=info, run=run,
                                           callback_config=callback_config,
                                           verbose=1)
                           for run, csc_kwargs in enumerate(kwargs_grid))

        info['grid_key'] = list(exp.grid.keys())

    else:
        res = [run_one_cached(
            X, default_kwargs, n_jobs=args.n_jobs, info=info,
            callback_config=callback_config, verbose=1)]
        kwargs_grid = [exp.default_kwargs]
        info['grid_key'] = ['reg']

    try:
        data = [(args, d) for args, d in zip(kwargs_grid, res)]

        print("Save the results")
        import pickle
        with open("{}/data.pkl".format(dirname), "wb") as f:
            pickle.dump(data, f)

        print("Plotting output")
        plt.close('all')
        for key, plot in PLOTS.items():
            if key in output_config:
                plot(data, info, dirname)
    finally:
        # Delete the directory if in debug mode
        if args.debug:
            shutil.rmtree(dirname)

    # Display the outputs figures
    plt.show()

    IPython.embed()
