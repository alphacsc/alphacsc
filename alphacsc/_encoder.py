import numpy as np
from .loss_and_gradient import compute_X_and_objective_multi
from .update_z_multi import update_z_multi
from .utils import check_dimension, lil


# XXX check consistency / proper use!
def get_z_encoder_for(
        solver,
        z_kwargs,
        X,
        D_hat,
        n_atoms,
        atom_support,
        algorithm,
        reg,
        loss,
        loss_params,
        uv_constraint,
        feasible_evaluation,
        n_jobs,
        use_sparse_z):
    """
    Returns a z encoder for the required solver.
    Allowed solvers are ['l-bfgs', 'lgcd']

    Parameters
    ----------
    algorithm : 'batch' | 'greedy' | 'online' | 'stochastic'
        Dictionary learning algorithm.

    Returns
    -------
    enc - a ZEncoder instance

    Example usage
    -------------
    with get_encoder_for('lgcd') as enc:
        ...
    """
    if solver in ['l-bfgs', 'lgcd']:
        return AlphaCSCEncoder(
            solver,
            z_kwargs,
            X,
            D_hat,
            n_atoms,
            atom_support,
            algorithm,
            reg,
            loss,
            loss_params,
            uv_constraint,
            feasible_evaluation,
            n_jobs,
            use_sparse_z)
    else:
        raise ValueError(f'unrecognized solver type: {solver}.')


class BaseZEncoder:
    def compute_z(self):
        """
        Perform one incremental z update.
        This is the "main" function of the algorithm.
        """
        raise NotImplementedError()

    def compute_z_partial(self, i0):
        """
        Compute z on a slice of the signal X,
        for online learning.

        Parameters
        ----------
        i0 : int
             Slice index.
        """
        raise NotImplementedError()

    def get_cost(self):
        """
        Computes the cost of the current sparse representation (z_hat)

        Returns
        -------
        cost: float
        """
        raise NotImplementedError()

    def get_sufficient_statistics(self):
        """
        Computes sufficient statistics to update D.

        Returns
        -------
        ztz, ztX : (float, float)
            Sufficient statistics.
        """
        raise NotImplementedError()

    def get_sufficient_statistics_partial(self):
        """
        Returns the partial sufficient statistics
        that were computed during the last call to
        compute_z_partial.

        Returns
        -------
        ztz, ztX : (float, float)
            Sufficient statistics for the slice that was
            selected in the last call of ``compute_z_partial``
        """
        raise NotImplementedError()

    def get_z_hat(self):
        """
        Returns a sparse encoding of the signal.

        Returns
        -------
        z_hat
            Sparse encoding of the signal X.
        """
        raise NotImplementedError()

    def set_D(self, d):
        """
        Update the dictionary.

        Parameters
        ----------
        d
            An updated dictionary, to be used for the next
            computation of z_hat.
        """
        raise NotImplementedError()

    def set_reg(self, reg):
        """
        Update the regularization parameter.

        Parameters
        ----------
        reg : float
              Regularization parameter
        """
        raise NotImplementedError()

    def add_one_atom(self, new_atom):
        """
        Add one atom to the dictionary and extend z_hat
        to match the new dimensions.

        Parameters
        ----------
        new_atom : array, shape (n_channels + n_times_atom)
            A new atom to add to the dictionary.
        """
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class AlphaCSCEncoder(BaseZEncoder):
    def __init__(
            self,
            solver,
            z_kwargs,
            X,
            D_hat,
            n_atoms,
            atom_support,
            algorithm,
            reg,
            loss,
            loss_params,
            uv_constraint,
            feasible_evaluation,
            n_jobs,
            use_sparse_z):
        self.z_alg = solver
        self.z_kwargs = z_kwargs or dict()
        self.X = X
        self.D_hat = D_hat
        self.n_atoms = n_atoms
        self.atom_support = atom_support
        self.algorithm = algorithm
        self.reg = reg or 0.1
        self.loss = loss
        self.loss_params = loss_params
        self.uv_constraint = uv_constraint
        self.feasible_evaluation = feasible_evaluation
        self.n_jobs = n_jobs
        self.use_sparse_z = use_sparse_z

        self._init_z_hat()

    def _init_z_hat(self):
        n_trials, _, n_times = check_dimension(self.X)
        n_times_valid = n_times - self.atom_support + 1

        self.z_hat = lil.init_zeros(
            self.use_sparse_z, n_trials, self.n_atoms, n_times_valid)

        if self.algorithm == 'greedy':
            # remove all atoms
            self.D_hat = self.D_hat[0:0]
            # remove all activations
            use_sparse_z = lil.is_list_of_lil(self.z_hat)
            n_trials, _, n_times_valid = lil.get_z_shape(self.z_hat)
            self.z_hat = lil.init_zeros(
                use_sparse_z, n_trials, 0, n_times_valid)

    def _compute_z_aux(self, X, z0, unbiased_z_hat):
        reg = self.reg if not unbiased_z_hat else 0

        return update_z_multi(
            X,
            self.D_hat,
            reg=reg,
            z0=z0,
            solver=self.z_alg,
            solver_kwargs=self.z_kwargs,
            freeze_support=unbiased_z_hat,
            loss=self.loss,
            loss_params=self.loss_params,
            n_jobs=self.n_jobs,
            return_ztz=True)

    def compute_z(self, unbiased_z_hat=False):
        self.z_hat, self.ztz, self.ztX = self._compute_z_aux(
            self.X, self.z_hat, unbiased_z_hat)

    def compute_z_partial(self, i0):
        self.z_hat[i0], self.ztz_i0, self.ztX_i0 = self._compute_z_aux(
            self.X[i0], self.z_hat[i0], unbiased_z_hat=False)

    def get_cost(self):
        cost = compute_X_and_objective_multi(self.X, self.z_hat, self.D_hat,
                                             reg=self.reg, loss=self.loss,
                                             loss_params=self.loss_params,
                                             uv_constraint=self.uv_constraint,
                                             feasible_evaluation=True,
                                             return_X_hat=False)
        return cost

    def get_sufficient_statistics(self):
        return self.ztz, self.ztX

    def get_sufficient_statistics_partial(self):
        return self.ztz_i0, self.ztX_i0

    def set_D(self, D):
        self.D_hat = D

    def set_reg(self, reg):
        self.reg = reg

    def add_one_atom(self, new_atom):
        self.D_hat = np.concatenate([self.D_hat, new_atom[None]])
        self.z_hat = lil.add_one_atom_in_z(self.z_hat)

    def get_z_hat(self):
        return self.z_hat
