
dataset = 'simu'

sfreq = 300

dataset_kwargs = dict(
    n_trials=30, n_channels=1, T=4.,
    sfreq=sfreq, sigma=.5, random_state=42
)


n_times_atom = int(sfreq * .5)  # .5 sec

csc_kwargs = dict(
    n_atoms=10,
    n_times_atom=n_times_atom,
    n_iter=2,
    reg=.5,
    eps=1e-3,
    solver_z="gcd",
    solver_z_kwargs={'max_iter': 1000},
    solver_d='fista',
    solver_d_kwargs={'max_iter': 100},
    D_init='ssa',
    D_init_params=dict(distances='trans_inv'),
    uv_constraint='separate',
    rank1=True,
    loss='l2',
    loss_params=dict(gamma=0.005, sakoe_chiba_band=10),
    algorithm='batch',
    lmbd_max='shared',
    random_state=27,
)

# Callback config display atom shape and KDE of z_hat
config = {
    'atom': {},
    'z_hat': {}
}
