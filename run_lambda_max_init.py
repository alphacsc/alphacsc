import numpy as np

from multicsc.datasets.somato import load_data
from multicsc.utils.dictionary import get_lambda_max
from multicsc.init_dict import init_dictionary
from multicsc.learn_d_z_multi import learn_d_z_multi

X, info = load_data(epoch=False, n_trials=2)
X = X[:, :, :10000]

n_atoms, n_times_atom = 8, 32
n_trials, n_channels, n_times = X.shape
n_times_valid = n_times - n_times_atom + 1

for D_init in ['kmeans', 'ssa', 'chunk', 'random']:
    print('')
    for random_state in range(3):
        D_hat = init_dictionary(X, n_atoms, n_times_atom, D_init=D_init,
                                rank1=True, uv_constraint='separate',
                                kmeans_params=dict(),
                                random_state=random_state)
        lambda_max = get_lambda_max(X, D_hat).max()
        print(D_init, random_state, lambda_max)

        z_hat = np.zeros((n_atoms, n_trials, n_times_valid))
        for i in range(2):
            pobj, times, D_hat, z_hat = learn_d_z_multi(
                X, n_atoms, n_times_atom, reg=lambda_max / 10., n_iter=1,
                D_init=D_hat, verbose=0, n_jobs=2)
            lambda_max = get_lambda_max(X, D_hat).max()
            print('           ', lambda_max)
