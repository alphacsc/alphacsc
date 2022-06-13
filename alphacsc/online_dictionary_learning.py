import numpy as np

from .utils.dictionary import get_lambda_max

from .convolutional_dictionary_learning import DOC_FMT, DEFAULT
from .convolutional_dictionary_learning import ConvolutionalDictionaryLearning

from ._z_encoder import get_z_encoder_for
from ._d_solver import get_solver_d


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
                 solver_d='auto', solver_d_kwargs={}, rank1=True, window=False,
                 uv_constraint='auto', lmbd_max='scaled', eps=1e-10,
                 D_init=None, alpha=.8, batch_size=1,
                 batch_selection='random', verbose=10, random_state=None):
        super().__init__(
            n_atoms, n_times_atom, reg=reg, n_iter=n_iter,
            solver_z=solver_z, solver_z_kwargs=solver_z_kwargs,
            rank1=rank1, window=window, uv_constraint=uv_constraint,
            unbiased_z_hat=unbiased_z_hat,
            solver_d=solver_d, solver_d_kwargs=solver_d_kwargs,
            eps=eps, D_init=D_init,
            algorithm_params=dict(alpha=alpha, batch_size=batch_size,
                                  batch_selection=batch_selection),
            n_jobs=n_jobs, random_state=random_state, algorithm='online',
            lmbd_max=lmbd_max, raise_on_increase=False,
            callback=None, verbose=verbose, name="OnlineCDL"
        )
        self.index = 0

    def partial_fit(self, X, y=None):
        # Successive partial_fit are equivalent to OnlineCDL only if
        # the X passed to this method are taken from a normalized
        # X_full ( X_full / X_full.std())
        self._ensure_fit_init(X)

        with get_z_encoder_for(X, self._D_hat, self.n_atoms, self.n_times_atom,
                               self.n_jobs, self.solver_z,
                               self.solver_z_kwargs, self.reg_) as z_encoder:

            z_encoder.compute_z()

            alpha = self.algorithm_params['alpha']
            self.constants['ztz'] *= alpha
            self.constants['ztz'] += z_encoder.ztz
            z_encoder.ztz = self.constants['ztz']
            self.constants['ztX'] *= alpha
            self.constants['ztX'] += z_encoder.ztX
            z_encoder.ztX = self.constants['ztX']
            self.constants['XtX'] *= alpha
            self.constants['XtX'] += z_encoder.XtX
            z_encoder.XtX = self.constants['XtX']

            z_nnz = z_encoder.get_z_nnz()

            if self.verbose > 5:
                print("[{}] sparsity: {:.3e}".format(
                    self.name, z_nnz.mean()))

            if np.all(z_nnz == 0):
                # No need to update the dictionary as this batch has all its
                # activation to 0.
                import warnings
                warnings.warn("Regularization parameter `reg` is too large and"
                              " all the activations are zero. The atoms have"
                              " not been updated.", UserWarning)
                return z_encoder.get_z_hat()

            self._D_hat = self.d_solver.update_D(z_encoder)
            self.z_hat = z_encoder.get_z_hat()

            return self.z_hat

    def _ensure_fit_init(self, X):
        """Initialization for p partial_fit."""

        assert X.ndim == 3

        if hasattr(self, 'constants'):
            return

        self.constants = {}
        self.n_channels_ = n_channels = X.shape[1]

        self.constants['n_channels'] = n_channels
        self.constants['XtX'] = 0

        self.constants['ztz'] = np.zeros((self.n_atoms, self.n_atoms,
                                          2 * self.n_times_atom - 1))
        self.constants['ztX'] = np.zeros((self.n_atoms, n_channels,
                                          self. n_times_atom))

        self.d_solver = get_solver_d(
            n_channels, self.n_atoms, self.n_times_atom,
            solver_d=self.solver_d, rank1=self.rank1, window=self.window,
            D_init=self.D_init, random_state=self.random_state,
            **self.solver_d_kwargs
        )

        # Init dictionary either from D_init or from an heuristic based on the
        # first batch X
        self._D_hat = self.d_solver.init_dictionary(X)

        self.reg_ = self.reg
        _lmbd_max = get_lambda_max(X, self._D_hat).max()
        if self.lmbd_max == "scaled":
            self.reg_ *= _lmbd_max
