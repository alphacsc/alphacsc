import numpy as np


from .init_dict import init_dictionary
from .update_z_multi import update_z_multi
from .utils.dictionary import get_lambda_max
from .update_d_multi import update_d, update_uv

from .convolutional_dictionary_learning import DOC_FMT, DEFAULT
from .convolutional_dictionary_learning import ConvolutionalDictionaryLearning


class OnlineCDL(ConvolutionalDictionaryLearning):
    _default = {}
    _default.update(DEFAULT)
    _default['desc'] = "Online algorithm for convolutional dictionary learning"
    _default['algorithm'] = """    Online algorithm

    alpha : float
        Forgetting factor for online learning. If set to 0, the learning is
        stochastic and each D-step is independent from the previous steps.
        When set to 1, each the previous values z_hat - computed with different
        dictionary - have the same weight as the current one. This factor
        should be large enough to ensure convergence but to large factor can
        lead to sub-optimal minima.
    batch_selection : 'random' | 'cyclic'
        The batch selection strategy for online learning. The batch are either
        selected randomly among all samples (without replacement) or in a
        cyclic way.
    batch_size : int in [1, n_trials]
        Size of the batch used in online learning. Increasing it regularizes
        the dictionary learning as there is less variance for the successive
        estimates. But it also increases the computational cost as more coding
        signals z_hat must be estimate at each iteration.
    """
    __doc__ = DOC_FMT.format(**_default)

    def __init__(self, n_atoms, n_times_atom, reg=0.1, n_iter=60, n_jobs=1,
                 solver_z='lgcd', solver_z_kwargs={}, unbiased_z_hat=False,
                 solver_d='alternate_adaptive', solver_d_kwargs={},
                 rank1=True, window=False, uv_constraint='separate',
                 lmbd_max='scaled', eps=1e-10, D_init=None, D_init_params={},
                 alpha=.8, batch_size=1, batch_selection='random',
                 verbose=10, random_state=None):
        super().__init__(
            n_atoms, n_times_atom, reg=reg, n_iter=n_iter,
            solver_z=solver_z, solver_z_kwargs=solver_z_kwargs,
            rank1=rank1, window=window, uv_constraint=uv_constraint,
            unbiased_z_hat=unbiased_z_hat,
            solver_d=solver_d, solver_d_kwargs=solver_d_kwargs,
            eps=eps, D_init=D_init, D_init_params=D_init_params,
            algorithm_params=dict(alpha=alpha, batch_size=batch_size,
                                  batch_selection=batch_selection),
            n_jobs=n_jobs, random_state=random_state, algorithm='online',
            lmbd_max=lmbd_max, raise_on_increase=False, loss='l2',
            callback=None, use_sparse_z=False, verbose=verbose,
            name="OnlineCDL")

    def partial_fit(self, X, y=None):
        self._check_param_partial_fit()
        self._ensure_fit_init(X)

        # Compute the activations for the current batch and get the sufficient
        # statistic for the dictionary update
        z_hat, ztz, ztX = update_z_multi(
            X, self._D_hat, reg=self.reg_,
            solver=self.solver_z, solver_kwargs=self.solver_z_kwargs,
            loss=self.loss, loss_params=self.loss_params,
            n_jobs=self.n_jobs, return_ztz=True)

        alpha = self.algorithm_params['alpha']
        self.constants['XtX'] = X.ravel().dot(X.ravel())
        self.constants['ztz'] = alpha * self.constants['ztz'] + ztz
        self.constants['ztX'] = alpha * self.constants['ztX'] + ztX

        # Make sure the activation is not all 0
        z_nnz = np.sum(z_hat != 0, axis=(0, 2))

        if self.verbose > 5:
            print("[{}] sparsity: {:.3e}".format(
                self.name, z_nnz.sum() / z_hat.size))

        if np.all(z_nnz == 0):
            # No need to update the dictionary as this batch has all its
            # activation to 0.
            import warnings
            warnings.warn("Regularization parameter `reg` is too large and all"
                          " the activations are zero. The atoms has not been "
                          "updated.", UserWarning)
            return z_hat

        d_kwargs = dict(verbose=self.verbose, eps=1e-8)
        d_kwargs.update(self.solver_d_kwargs)
        if self.rank1:
            self._D_hat = update_uv(
                X, z_hat, uv_hat0=self._D_hat, constants=self.constants,
                solver_d=self.solver_d, uv_constraint=self.uv_constraint,
                loss=self.loss, loss_params=self.loss_params,
                window=self.window, **d_kwargs)
        else:
            self._D_hat = update_d(
                X, z_hat, D_hat0=self._D_hat, constants=self.constants,
                solver_d=self.solver_d, uv_constraint=self.uv_constraint,
                loss=self.loss, loss_params=self.loss_params,
                window=self.window, **d_kwargs)

        return z_hat

    def _ensure_fit_init(self, X):
        """Initialization for p partial_fit."""

        assert X.ndim == 3

        if hasattr(self, 'constants'):
            return

        self.constants = {}
        self.n_channels_ = n_channels = X.shape[1]

        self.constants['n_channels'] = n_channels
        self.constants['ztz'] = np.zeros((self.n_atoms, self.n_atoms,
                                          2 * self.n_times_atom - 1))
        self.constants['ztX'] = np.zeros((self.n_atoms, n_channels,
                                          self. n_times_atom))

        # Init dictionary either from D_init or from an heuristic based on the
        # first batch X
        self._D_hat = init_dictionary(
            X, self.n_atoms, self.n_times_atom, rank1=self.rank1,
            window=self.window, uv_constraint=self.uv_constraint,
            D_init=self.D_init, D_init_params=self.D_init_params,
            random_state=self.random_state)

        self.reg_ = self.reg
        _lmbd_max = get_lambda_max(X, self._D_hat).max()
        if self.lmbd_max == "scaled":
            self.reg_ *= _lmbd_max

    def _check_param_partial_fit(self):
        assert self.loss == 'l2', (
            "partial_fit is implemented only for loss={}.".format(self.loss))
