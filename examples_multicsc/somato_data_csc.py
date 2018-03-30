import IPython

from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.utils.viz import get_callback_csc, DEFAULT_CB
from alphacsc.datasets.somato import load_data

n_jobs = 4
reg = 11
sfreq = 300
n_atoms = 10
random_state = 42

X, info = load_data(sfreq=sfreq, n_jobs=n_jobs, epoch=True)
n_times_atom = int(sfreq * .5)

csc_kwargs = dict(
    n_atoms=n_atoms, n_times_atom=n_times_atom,
    random_state=random_state, n_iter=50, reg=reg, eps=1e-3,
    solver_z="gcd", solver_z_kwargs={'max_iter': 1000},
    solver_d='alternate_adaptive', solver_d_kwargs={'max_iter': 100},
    D_init='kmeans', kmeans_params=dict(distances='trans_inv'),
    uv_constraint='separate', loss='l2',
    loss_params=dict(gamma=0.005, sakoe_chiba_band=10),
    algorithm='batch', lmbd_max='fixed'
)

# Callback config display atom shape and KDE of Z_hat
config = DEFAULT_CB

callback = get_callback_csc(csc_kwargs, config=config, info=info)


try:
    pobj, times, uv_hat, Z_hat = learn_d_z_multi(
        X, n_jobs=n_jobs, callback=callback,
        **csc_kwargs)
except KeyboardInterrupt:
    pass

IPython.embed()
