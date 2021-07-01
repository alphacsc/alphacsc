import warnings

from .loss_and_gradient import compute_X_and_objective_multi
from .update_z_multi import update_z_multi

# XXX needs additional args(signal, options...)
# -> Either here (maybe it's simpler for a start?) or
# in a separate `ZEncoder.init_encoder` method?
#
# At minimum: z solver, X, D_hat, reg.
# AlphaCSC backend: loss, loss_params, uv_constraint, feasible_evaluation
# DiCoDiLe backend: n_workers
# XXX check consistency / proper use!
# XXX reg is updated during descent steps... add a method on the encoder?
# XXX solver_kwargs!!


def get_z_encoder_for(solver, z_kwargs, X, z_hat, D_hat, reg, loss, loss_params, uv_constraint, feasible_evaluation, return_X_hat):
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
        # n_workers should be user-provided (distributed, cannot set to cpu count...)
        # -> how?
        return DicodileEncoder(X, n_workers=10)  # XXX n_workers
    elif solver in ['l-bfgs', 'lgcd']:
        return AlphaCSCEncoder(solver, z_kwargs, X, z_hat, D_hat, reg, loss, loss_params, uv_constraint, feasible_evaluation, return_X_hat)
    else:
        raise ValueError(f'unrecognized solver type: {solver}.')


class ZEncoder:
    def update_z(self):
        """
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
        """
        """
        raise NotImplementedError()

    def set_D(self, d):
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DicodileEncoder(ZEncoder):
    def __init__(self, X, D_hat, reg, n_workers):
        try:
            from dicodile.update_z.distributed_sparse_encoder import DistributedSparseEncoder
        except ImportError as ie:
            warnings.warn(
                'Please install dicodile by running "pip install alphacsc[dicodile]"')
            raise
        self.encoder = DistributedSparseEncoder(n_workers)
        # perform init steps (send X,D...)
        # XXX do we need to resend reg at some point?
        self.encoder.init_workers(X, D_hat, reg)

    def update_z(self):
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


class AlphaCSCEncoder(ZEncoder):
    def __init__(self, solver, z_kwargs, X, z_hat, D_hat, reg, loss, loss_params, uv_constraint, feasible_evaluation, return_X_hat):
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
        self.return_X_hat = return_X_hat

    def update_z(self):
        # XXX missing params!!!
        self.z_hat, self.ztz, self.ztX = update_z_multi(
            self.X, self.D_hat, reg=self.reg, z0=self.z_hat, solver=self.z_alg,
            solver_kwargs=self.z_kwargs, loss=self.loss, loss_params=self.loss_params, n_jobs=1, return_ztz=True) #XXX solver_kwargs! XXX n_jobs!

    def get_cost(self):
        cost = compute_X_and_objective_multi(self.X, self.z_hat, self.D_hat,
                                             reg=self.reg, loss=self.loss,
                                             loss_params=self.loss_params,
                                             uv_constraint=self.uv_constraint,
                                             feasible_evaluation=True,
                                             return_X_hat=self.return_X_hat)
        return cost

    def get_sufficient_statistics(self):
        return self.ztz, self.ztX

    def set_D(self, D):
        self.D_hat = D

    def get_z_hat(self):
        return self.z_hat
