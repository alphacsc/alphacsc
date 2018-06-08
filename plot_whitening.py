import matplotlib.pyplot as plt

from multicsc.learn_d_z_multi import learn_d_z_multi
from multicsc.utils.whitening import whitening, unwhitening
from multicsc.utils.viz import get_callback_csc, plot_or_replot

###############################################################################
if True:
    # somato dataset
    from multicsc.datasets.somato import load_data
    print('loading the data...')
    sfreq = 150.
    X, info = load_data(sfreq=sfreq, filt=[None, None])
    n_trials, n_channels, n_times = X.shape

    csc_kwargs = dict(n_times_atom=int(round(sfreq * 0.3)), n_atoms=10, reg=10)

else:
    # simulation
    from multicsc.datasets.simulate import load_data
    print('creating the data...')
    sfreq = 300.
    n_times = 1024
    n_trials = 20
    n_channels = 1

    X, info = load_data(n_trials=n_trials, n_channels=n_channels,
                        T=n_times / sfreq, sigma=.1, sfreq=sfreq,
                        f_noise=True, random_state=None)

    csc_kwargs = dict(n_times_atom=int(round(sfreq * 0.3)), n_atoms=2,
                      reg=0.1)

###############################################################################
# whitening
print('whitening...')
use_fooof = True
ar_model, X_white = whitening(X, ordar=8, sfreq=sfreq, plot=True,
                              use_fooof=use_fooof)

plt.figure()
plt.subplot(211)
plt.plot(X[0, 0], label='X')
plt.legend()
plt.subplot(212)
plt.plot(X_white[0, 0], c='C2', label='X_white')
plt.legend()
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
plot_or_replot(v_hat_unwhite)
plt.show()
