import numpy as np
from .loss_and_gradient import compute_X_and_objective_multi
from .update_z_multi import update_z_multi
from .utils import check_dimension, lil

# XXX check consistency / proper use!
def get_z_encoder_for(solver, z_kwargs, X, D_hat, n_atoms, atom_support, algorithm, reg, loss, loss_params, uv_constraint, feasible_evaluation, n_jobs):
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
        return AlphaCSCEncoder(solver, z_kwargs, X, D_hat, n_atoms, atom_support, algorithm, reg, loss, loss_params, uv_constraint, feasible_evaluation, n_jobs)
    else:
        raise ValueError(f'unrecognized solver type: {solver}.')


class BaseZEncoder:
    def compute_z(self):
        """
        """
        raise NotImplementedError()
    
    def compute_z_partial(self, i0):
        """
        (Online learning)
        """
        raise NotImplementedError()

    def get_cost(self):
        """
        Returns
        -------
        cost: float
        """
        raise NotImplementedError()

    def get_sufficient_statistics(self):
        """
        Returns
        -------
        ztz, ztX
        """
        raise NotImplementedError()

    def get_z_hat(self):
        raise NotImplementedError()
    
    def get_z_hat_partial(self, i0):
        """
        (Online learning)
        """
        raise NotImplementedError()

    def set_D(self, d):
        """
        Update the dictionary
        """
        raise NotImplementedError()
    
    def set_reg(self, reg):
        """
        Update the regularization parameter
        """
        raise NotImplementedError()
    
    def add_one_atom(self, new_atom):
        """
        (Greedy learning)
        """
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class AlphaCSCEncoder(BaseZEncoder):
    def __init__(self, solver, z_kwargs, X, D_hat, n_atoms, atom_support, algorithm, reg, loss, loss_params, uv_constraint, feasible_evaluation, n_jobs):
        self.z_alg = solver 
        self.z_kwargs = z_kwargs
        self.X = X
        self.D_hat = D_hat
        self.n_atoms = n_atoms
        self.atom_support = atom_support
        self.algorithm = algorithm
        self.reg = reg
        self.loss = loss
        self.loss_params = loss_params
        self.uv_constraint = uv_constraint
        self.feasible_evaluation = feasible_evaluation
        self.n_jobs = n_jobs

        self._init_z_hat()
    
    def _init_z_hat(self):
        n_trials, _, n_times = check_dimension(self.X)
        n_times_valid = n_times - self.atom_support + 1

        self.z_hat = lil.init_zeros(False, n_trials, self.n_atoms, n_times_valid) #XXX use_sparse_z forced to False


        if self.algorithm == 'greedy':
            # remove all atoms
            self.D_hat = self.D_hat[0:0]
            # remove all activations
            use_sparse_z = lil.is_list_of_lil(self.z_hat)
            n_trials, _, n_times_valid = lil.get_z_shape(self.z_hat)
            self.z_hat = lil.init_zeros(use_sparse_z, n_trials, 0, n_times_valid)
    
    def _compute_z_aux(self, X, z0):
        return update_z_multi(
            X, self.D_hat, reg=self.reg, z0=z0, solver=self.z_alg,
            solver_kwargs=self.z_kwargs, loss=self.loss, loss_params=self.loss_params, n_jobs=self.n_jobs, return_ztz=True)
    
    def compute_z(self):
        self.z_hat, self.ztz, self.ztX = self._compute_z_aux(
            self.X, self.z_hat)

    def compute_z_partial(self, i0):
        self.z_hat[i0], self.ztz_i0, self.ztX_i0 = self._compute_z_aux( #XXX
            self.X[i0], self.z_hat[i0])

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
        """
        Compute the partial sufficient statistics
        that were computed during the last call to 
        compute_z_partial
        """
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
