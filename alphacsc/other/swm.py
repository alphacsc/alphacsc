"""
Code adopted from Voytek Lab package neurodsp:
https://github.com/voytekresearch/neurodsp/blob/master/neurodsp/shape/swm.py

The sliding window matching algorithm identifies the waveform shape of
neural oscillations using correlations.
"""

# Authors: Scott Cole
#          Mainak Jas <mainak.jas@telecom-paristech.fr>

import numpy as np
from scipy.spatial.distance import pdist

from .utils import check_random_state


def sliding_window_matching(x, L, G, max_iterations=500, T=1,
                            window_starts_custom=None, random_state=None):
    """Find recurring patterns in a time series using SWM algorithm.

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    L : float
        window length (seconds)
    G : float
        minimum window spacing (seconds)
    T : float
        temperature parameter. Controls probability of accepting a new window
    max_iterations : int
        Maximum number of iterations of potential changes in window placement
    window_starts_custom : np.ndarray (1d)
        Pre-set locations of initial windows (instead of evenly spaced by 2G)
    random_state : int
        The random state

    Returns
    -------
    avg_window : ndarray (1d)
        The average waveform in x.
    window_starts : ndarray (1d)
        Indices at which each window begins for the final set of windows
    J : np.ndarray (1d)
        Cost function value at each iteration

    References
    ----------
    Gips, B., Bahramisharif, A., Lowet, E., Roberts, M. J., de Weerd, P.,
    Jensen, O., & van der Eerden, J. (2017). Discovering recurring
    patterns in electrophysiological recordings.
    Journal of Neuroscience Methods, 275, 66-79.
    MATLAB code: https://github.com/bartgips/SWM

    Notes
    -----
    * Apply a highpass filter if looking at high frequency activity,
      so that it does not converge on a low frequency motif
    * L and G should be chosen to be about the size of the motif of interest
    """
    rng = check_random_state(random_state)

    # Initialize window positions, separated by 2*G
    if window_starts_custom is None:
        window_starts = np.arange(0, len(x) - L, 2 * G)
    else:
        window_starts = window_starts_custom
    N_windows = len(window_starts)

    # Calculate initial cost
    J = np.zeros(max_iterations)
    J[0] = _compute_J(x, window_starts, L)

    # Randomly sample windows with replacement
    random_window_idx = rng.choice(range(N_windows), size=max_iterations)

    # For each iteration, randomly replace a window with a new window
    # to improve cross-window similarity
    for idx in range(1, max_iterations):
        # Pick a random window position
        window_idx_replace = random_window_idx[idx]

        # Find a new allowed position for the window
        window_starts_temp = np.copy(window_starts)
        window_starts_temp[window_idx_replace] = _find_new_windowidx(
            window_starts, G, L, len(x) - L, rng)

        # Calculate the cost with replaced windows
        J_temp = _compute_J(x, window_starts_temp, L)

        # Calculate the change in cost function
        deltaJ = J_temp - J[idx - 1]

        # Calculate the acceptance probability
        p_accept = np.exp(-deltaJ / float(T))

        # Accept update to J with a certain probability
        if rng.rand() < p_accept:
            J[idx] = J_temp
            # Update X
            window_starts = window_starts_temp
        else:
            J[idx] = J[idx - 1]

        print('[iter %03d] Cost function: %s' % (idx, J[idx]))

    # Calculate average window
    avg_window = np.zeros(L)
    for w in range(N_windows):
        avg_window += x[window_starts[w]:window_starts[w] + L]
    avg_window = avg_window / float(N_windows)

    return avg_window, window_starts, J


def _compute_J(x, window_starts, L):
    """Compute the cost, which is proportional to the
    difference between pairs of windows"""

    # Get all windows and zscore them
    N_windows = len(window_starts)
    windows = np.zeros((N_windows, L))
    for w in range(N_windows):
        temp = x[window_starts[w]:window_starts[w] + L]
        windows[w] = (temp - np.mean(temp)) / np.std(temp)

    # Calculate distances for all pairs of windows
    dist = pdist(np.vstack(windows),
                 lambda u, v: np.sum((u - v) ** 2))
    J = np.sum(dist) / float(L * (N_windows - 1))

    return J


def _find_new_windowidx(window_starts, G, L, N_samp, rng,
                        tries_limit=1000):
    """Find a new sample for the starting window"""

    found = False
    N_tries = 0
    while found is False:
        # Generate a random sample
        new_samp = rng.randint(N_samp)
        # Check how close the sample is to other window starts
        dists = np.abs(window_starts - new_samp)
        if np.min(dists) > G:
            return new_samp
        else:
            N_tries += 1
            if N_tries > tries_limit:
                raise RuntimeError('SWM algorithm has difficulty finding a new'
                                   ' window. Increase the spacing parameter,'
                                   ' G.')
