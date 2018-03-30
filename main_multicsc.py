import argparse

from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.utils.viz import get_callback_csc


DATASETS = ['simu']


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Main script for multi csc experiements.')
    parser.add_argument('--n_iter', type=int, default=50,
                        help='Print profiling of the function')
    parser.add_argument('--dataset', type=str, default='simu',
                        help='Dataset used in the experiment.')

    args = parser.parse_args()

    reg = .05
    sfreq = 300
    n_atoms = 10
    n_trials = 30
    n_channels = 1
    random_state = 42

    if args.dataset == 'simu':
        from alphacsc.datasets.simulate import load_data
        X, info = load_data(n_trials=n_trials, n_channels=n_channels, T=4.,
                            sfreq=sfreq, sigma=.5, random_state=random_state)
    else:
        raise NotImplementedError("No dataset named {}. Should select one in "
                                  "{}".format(args.dataset, DATASETS))

    n_times_atom = int(sfreq * .2)  # .5 sec

    csc_kwargs = dict(
        n_atoms=n_atoms, n_times_atom=n_times_atom,
        random_state=random_state, n_iter=args.n_iter, reg=reg, eps=1e-3,
        solver_z="gcd", solver_z_kwargs={'max_iter': 1000},
        solver_d='fista', solver_d_kwargs={'max_iter': 100},
        D_init='ssa', kmeans_params=dict(distances='trans_inv'),
        uv_constraint='separate', loss='l2', rank1=False,
        loss_params=dict(gamma=0.005, sakoe_chiba_band=10),
        algorithm='batch', lmbd_max='shared'
    )

    # Callback config display atom shape and KDE of Z_hat
    config = {
        'atom': {
            'rank1': False
        },
        'Zhat': {}
    }

    callback = get_callback_csc(csc_kwargs, info=info, config=config)

    pobj, times, uv_hat, Z_hat = learn_d_z_multi(
        X, n_jobs=1, name="MainCSC", verbose=5,
        callback=get_callback_csc(csc_kwargs, info=info, config=config),
        **csc_kwargs)
