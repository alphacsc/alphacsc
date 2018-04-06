import matplotlib.pyplot as plt

from alphacsc.learn_d_z_multi import learn_d_z_multi
from alphacsc.utils.whitening import whitening, unwhitening
from alphacsc.utils.viz import get_callback_csc

###############################################################################
if False:
    # somato dataset
    from alphacsc.datasets.somato import load_data
    print('loading the data...')
    sfreq = 150.
    X, info = load_data(sfreq=sfreq)
    n_trials, n_channels, n_times = X.shape

    csc_kwargs = dict(n_times_atom=int(round(sfreq * 0.3)), n_atoms=3, reg=6)

else:
    # simulation
    from alphacsc.datasets.simulate import load_data
    print('creating the data...')
    sfreq = 300.
    n_times = 1024
    n_trials = 20
    n_channels = 1

    X, info = load_data(n_trials=n_trials, n_channels=n_channels,
                        T=n_times / sfreq, sigma=.005, sfreq=sfreq,
                        f_noise=True, random_state=None)

    csc_kwargs = dict(n_times_atom=int(round(sfreq * 0.3)), n_atoms=2,
                      reg=0.03)

###############################################################################
# whitening
print('whitening...')
ar_model, X_white = whitening(X, sfreq=sfreq, plot=True)
plt.show()

###############################################################################
# fit the CSC model

n_iter = 50
random_state = 42

print('fitting CSC model...')
pobj, times, uv_hat, Z_hat = learn_d_z_multi(
    X=X_white,
    n_iter=n_iter,
    solver_d='joint',
    uv_constraint='separate',
    D_init='chunk',
    solver_z_kwargs={'factr': 1e10},
    random_state=random_state,
    n_jobs=1,
    verbose=1,
    callback=get_callback_csc(csc_kwargs),
    **csc_kwargs, )

###############################################################################
# un-whitening
print('unwhitening CSC model...')
v_hat = uv_hat[:, n_channels:]
v_hat_unwhite = unwhitening(ar_model, v_hat[None, :], plot=True)[0]

###############################################################################
plt.figure()
plt.plot(v_hat_unwhite.T)
plt.show()