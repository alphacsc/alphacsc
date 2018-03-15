import numpy as np
from scipy.signal import correlate
from scipy.linalg import eigh, blas

from .utils import check_random_state


def learn_atoms(X, n_atoms, n_times_atom, n_iter=10, max_shift=11,
                random_state=None):
    """Learn atoms using the MoTIF algorithm.

    Parameters
    ----------
    X : array, shape (n_trials, n_times)
        The data on which to apply MoTIF.
    n_atoms : int
        The number of atoms.
    n_times_atom : int
        The support of the atoms
    n_iter : int
        The number of iterations
    max_shift : int
        The maximum allowable shift for the atoms.
    random_state : int | None
        The random initialization.
    """

    rng = check_random_state(random_state)

    n_trials, n_times = X.shape

    atoms = rng.rand(n_atoms, n_times_atom)
    corrs = np.zeros(n_trials)
    match = np.zeros((n_atoms, n_trials), dtype=np.int)

    # loop through atoms
    for k in range(n_atoms):

        aligned_data = np.zeros((n_times_atom, n_trials))

        # compute Bk
        B = np.zeros((n_times_atom, n_times_atom), order='F')
        for l in range(k):
            for p in np.arange(max_shift):
                atom_shifted = np.roll(atoms[l], -p)[np.newaxis, :]
                # B += np.dot(atom_shifted.T, atom_shifted)
                B = blas.dger(1, atom_shifted, atom_shifted, a=B,
                              overwrite_a=1)

        # make B invertible by adding a full-rank matrix
        B += np.eye(B.shape[0]) * np.finfo(np.float32).eps

        for i in range(n_iter):
            print('[seed %s] Atom %d Iteration %d' % (random_state, k, i))
            # loop through training data
            for n in range(n_trials):
                # ### STEP 1: Find out where the data and atom align ####

                # which of these to use for template matching?
                vec1 = (X[n] - np.mean(X[n])) / (np.std(X[n]) * len(X[n]))
                vec2 = (atoms[k] - np.mean(atoms[k])) / np.std(atoms[k])
                tmp = np.abs(correlate(vec1, vec2, 'same'))

                offset = n_times_atom // 2
                match[k, n] = tmp[offset:-offset].argmax() + offset
                corrs[n] = tmp[match[k, n]]

                # aligned_data[:, n] = np.roll(X[n], -shift[n])[:n_times_atom]
                aligned_data[:, n] = X[n, match[k, n] - offset:
                                       match[k, n] + offset].copy()

            # ### STEP 2: Solve the generalized eigenvalue problem ####
            A = np.dot(aligned_data, aligned_data.T).copy()
            if k == 0:
                B = None
            e, U = eigh(A, B)
            # e, U = eigh(A)

            atoms[k, :] = U[:, -1] / np.linalg.norm(U[:, -1])
    return atoms
