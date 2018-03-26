import numpy as np

from .compat import jit


@jit
def _compute_DtD(uv, n_chan):
    """Compute the DtD matrix"""
    # TODO: benchmark the cross correlate function of numpy
    n_atoms, n_times_atom = uv.shape
    n_times_atom -= n_chan

    u = uv[:, :n_chan]
    v = uv[:, n_chan:]

    DtD = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    t0 = n_times_atom - 1
    for k0 in range(n_atoms):
        for k in range(n_atoms):
            for t in range(n_times_atom):
                if t == 0:
                    DtD[k0, k, t0] = np.dot(v[k0], v[k])
                else:
                    DtD[k0, k, t0 + t] = np.dot(v[k0, :-t], v[k, t:])
                    DtD[k0, k, t0 - t] = np.dot(v[k0, t:], v[k, :-t])
    DtD *= np.dot(u, u.T)[:, :, None]
    return DtD
