from .loss_and_gradient import compute_X_and_objective_multi
from .update_z_multi import update_z_multi
from .utils import lil

# XXX check consistency / proper use!
def get_z_encoder_for(solver, z_kwargs, X, z_hat, D_hat, reg, loss, loss_params, uv_constraint, feasible_evaluation, n_jobs):
    """
    Returns a z encoder for the required solver.
    Allowed solvers are ['l-bfgs', 'lgcd', dicodile']

    Parameters
    ----------

    Returns
    -------
    enc - a ZEncoder instance

    Example usage
    -------------
    with get_encoder_for('dicodile') as enc:
        ...
    """
    if solver == 'dicodile':
        return DicodileEncoder(X, n_workers=n_jobs) 
    elif solver in ['l-bfgs', 'lgcd']:
        return AlphaCSCEncoder(solver, z_kwargs, X, z_hat, D_hat, reg, loss, loss_params, uv_constraint, feasible_evaluation, n_jobs)
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
    
    def add_one_atom_in_z(self):
        """
        (Online learning)
        """
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DicodileEncoder(BaseZEncoder):
    def __init__(self, X, D_hat, reg, n_workers):
        try:
            from dicodile.update_z.distributed_sparse_encoder import DistributedSparseEncoder
        except ImportError as ie:
            raise ImportError('Please install dicodile by running "pip install alphacsc[dicodile]"') from ie
        self.encoder = DistributedSparseEncoder(n_workers)
        # perform init steps (send X,D...)
        self.encoder.init_workers(X, D_hat, reg)

    def compute_z(self):
        self.encoder.process_z_hat()

    def get_cost(self):
        return self.encoder.get_cost()

    def get_sufficient_statistics(self):
        return self.encoder.get_sufficient_statistics()

    def set_D(self, D):
        self.encoder.set_worker_D(D)

    def get_z_hat(self):
        return self.encoder.get_z_hat()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.encoder.release_workers()
        self.encoder.shutdown_workers()


class AlphaCSCEncoder(BaseZEncoder):
    def __init__(self, solver, z_kwargs, X, z_hat, D_hat, reg, loss, loss_params, uv_constraint, feasible_evaluation, n_jobs):
        self.z_alg = solver 
        self.z_kwargs = z_kwargs
        self.X = X
        self.z_hat = z_hat
        self.D_hat = D_hat
        self.reg = reg
        self.loss = loss
        self.loss_params = loss_params
        self.uv_constraint = uv_constraint
        self.feasible_evaluation = feasible_evaluation
        self.n_jobs = n_jobs
    
    def _compute_z_aux(self, X, z0):
        return update_z_multi(
            X, self.D_hat, reg=self.reg, z0=z0, solver=self.z_alg,
            solver_kwargs=self.z_kwargs, loss=self.loss, loss_params=self.loss_params, n_jobs=self.n_jobs, return_ztz=True)
    
    def compute_z(self):
        self.z_hat, self.ztz, self.ztX = self._compute_z_aux(
            self.X, self.z_hat)

    def compute_z_partial(self, i0):
        self.z_hat[i0], self.ztz, self.ztX = self._compute_z_aux(
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

    def set_D(self, D):
        self.D_hat = D
    
    def set_reg(self, reg):
        self.reg = reg
    
    def add_one_atom_in_z(self):
        self.z_hat = lil.add_one_atom_in_z(self.z_hat)

    def get_z_hat(self):
        return self.z_hat

    # XXX is that necessary?
    def get_z_hat_partial(self, i0):
        return self.z_hat[i0]