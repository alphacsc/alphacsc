import IPython

from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.utils.viz import get_callback_csc
from alphacsc.datasets.somato import load_data

n_jobs = 4
sfreq = 300

X = load_data(sfreq=sfreq, n_jobs=n_jobs)

csc_kwargs = dict(
    n_atoms=25, n_times_atom=int(sfreq * .2),
    random_state=42, n_iter=50, reg=.5, eps=1e-3,
    solver_z="gcd", solver_z_kwargs={'max_iter': 100},
    solver_d='alternate_adaptive', solver_d_kwargs={'max_iter': 100},
    uv_init='ssa', uv_constraint='separate', loss='l2',
    loss_params=dict(gamma=0.005, sakoe_chiba_band=10),
    algorithm='batch', lmbd_max='shared'
)

callback = get_callback_csc(csc_kwargs, sfreq)


try:
    pobj, times, uv_hat, Z_hat = learn_d_z_multi(
        X, n_jobs=n_jobs, callback=callback,
        **csc_kwargs)
except KeyboardInterrupt:
    pass

IPython.embed()
