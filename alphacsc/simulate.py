# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>

import numpy as np

from .utils import check_random_state, construct_X


def simulate_data(n_trials, n_times, n_times_atom, n_atoms, random_state=42,
                  constant_amplitude=False):
    """Simulate the data.

    Parameters
    ----------
    n_trials : int
        Number of samples / trials.
    n_times : int
        Number of time points.
    n_times_atom : int
        Number of time points.
    n_atoms : int
        Number of atoms.
    random_state : int | None
        If integer, fix the random state.
    constant_amplitude : float
        If True, the activations have constant amplitude.

    Returns
    -------
    X : array, shape (n_trials, n_times)
        The data
    ds : array, shape (k, n_times_atom)
        The true atoms.
    z : array, shape (n_trials, n_times - n_times_atom + 1)
        The true codes.

    Note
    ----
    X will be non-zero from n_times_atom to n_times.
    """

    starts = list()

    # add atoms
    rng = check_random_state(random_state)
    ds = np.zeros((n_atoms, n_times_atom))
    for idx, shape, n_cycles in cycler(n_atoms, n_times_atom):
        ds[idx, :] = get_atoms(shape, n_times_atom, n_cycles=n_cycles)
        starts.append(rng.randint(low=0, high=n_times - n_times_atom + 1,
                      size=(n_trials,)))
    ds /= np.linalg.norm(ds, axis=1)[:, None]

    # add activations
    Z = np.zeros((n_atoms, n_trials, n_times - n_times_atom + 1))
    for i in range(n_trials):
        for k_idx, (d, start) in enumerate(zip(ds, starts)):
            if constant_amplitude:
                randnum = 1.
            else:
                randnum = rng.uniform()
            Z[k_idx, i, starts[k_idx][i]] = randnum

    X = construct_X(Z, ds)

    assert X.shape == (n_trials, n_times)
    assert Z.shape == (n_atoms, n_trials, n_times - n_times_atom + 1)
    assert ds.shape == (n_atoms, n_times_atom)

    return X, ds, Z


def cycler(n_atoms, n_times_atom):
    idx = 0
    for n_cycles in range(1, n_times_atom // 2):
        for shape in ['triangle', 'square', 'sin']:
            yield idx, shape, n_cycles
            idx += 1
            if idx >= n_atoms:
                break
        if idx >= n_atoms:
            break


def get_atoms(shape, n_times_atom, zero_mean=True, n_cycles=1):
    if shape == 'triangle':
        ds = list()
        for idx in range(n_cycles):
            ds.append(np.linspace(0, 1, n_times_atom // (2 * n_cycles)))
            ds.append(ds[-1][::-1])
        d = np.hstack(ds)
        d = np.pad(d, (0, n_times_atom - d.shape[0]), 'constant')
    elif shape == 'square':
        ds = list()
        for idx in range(n_cycles):
            ds.append(0.5 * np.ones((n_times_atom // (2 * n_cycles))))
            ds.append(-ds[-1])
        d = np.hstack(ds)
        d = np.pad(d, (0, n_times_atom - d.shape[0]), 'constant')
    elif shape == 'sin':
        d = np.sin(2 * np.pi * n_cycles * np.linspace(0, 1, n_times_atom))
    if zero_mean:
        d -= np.mean(d)

    return d
