# MIT License
#
# Copyright (c) 2016 Olivier Bachem
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Fast and Provably Good Seedings for k-Means using k-MC2 and AFK-MC2

Cython implementation of the algorithms of the following two papers:

> Approximate K-Means++ in Sublinear Time
> Olivier Bachem, Mario Lucic, S. Hamed Hassani and Andreas Krause
> In Proc. Conference on Artificial Intelligence (AAAI), 2016.

> Fast and Provably Good Seedings for k-Means
> Olivier Bachem, Mario Lucic, S. Hamed Hassani and Andreas Krause
> To appear in Neural Information Processing Systems (NIPS), 2016.

Usage:
>>> import kmc2
>>> X = <Numpy array containing the data>
>>> seeding = kmc2.kmc2(X, 5)  # Run k-MC2 with k=5

Afterwards, we may run MiniBatchKMeans of Scikit-Learn:
>>> from sklearn.cluster import MiniBatchKMeans
>>> model = MiniBatchKMeans(5, init=seeding).fit(X)
>>> new_centers = model.cluster_centers_

Please refer to the doc string of kmc2.kmc2 for detailed usage.
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_array
from sklearn.utils.validation import check_random_state
cimport numpy as np
cimport cython

from custom_distances import roll_invariant_euclidean_distances
from custom_distances import translation_invariant_euclidean_distances


def kmc2(X, k, chain_length=200, afkmc2=True, random_state=None, weights=None,
         distances='euclidean'):
    """Cython implementation of k-MC2 and AFK-MC2 seeding

    Args:
      X: (n,d)-shaped np.ndarray with data points (or scipy CSR matrix)
      k: number of cluster centers
      chain_length: length of the MCMC chain
      afkmc2: Whether to run AFK-MC2 (if True) or vanilla K-MC2 (if False)
      random_state: numpy.random.RandomState instance or integer to be used as seed
      weights: n-sized np.ndarray with weights of data points (default: uniform weights)
      distances: in {'euclidean', 'roll_inv', 'trans_inv'}

    Returns:
      centers: (k, d)-shaped numpy.ndarray with cluster centers
      centers_indices: (k, )-shaped numpy.ndarray with cluster centers indices
    """
    # Local cython variables
    cdef np.intp_t j, curr_ind
    cdef double cand_prob, curr_prob
    cdef double[::1] q_cand, p_cand, rand_a

    # Handle input
    X = check_array(X, accept_sparse="csr", dtype=np.float64, order="C")
    sparse = not isinstance(X, np.ndarray)
    if weights is None:
        weights = np.ones(X.shape[0], dtype=np.float64)
    random_state = check_random_state(random_state)

    if distances == 'euclidean':
        distances = euclidean_distances
    elif distances == 'roll_inv':
        distances = roll_invariant_euclidean_distances
    elif distances == 'trans_inv':
        distances = translation_invariant_euclidean_distances
    else:
        raise ValueError('Unknown distances: "%s".' % (distances, ))

    # Initialize result
    centers_indices = np.zeros((k, ), np.intp)
    centers = np.zeros((k, X.shape[1]), np.float64, order="C")
    # Sample first center and compute proposal
    centers_indices[0] = random_state.choice(X.shape[0], p=weights/weights.sum())
    rel_row = X[centers_indices[0], :]
    centers[0, :] = rel_row.todense().flatten() if sparse else rel_row
    if afkmc2:
        di = np.min(distances(X, centers[0:1, :], squared=True), axis=1)*weights
        q = di/np.sum(di) + weights/np.sum(weights)  # Only the potentials
    else:
        q = np.copy(weights)
    # Renormalize the proposal distribution
    q = q / np.sum(q)

    for i in range(k-1):
        # Draw the candidate indices
        cand_ind = random_state.choice(X.shape[0], size=(chain_length), p=q).astype(np.intp)
        # Extract the proposal probabilities
        q_cand = q[cand_ind]
        # Compute pairwise distances
        dist = distances(X[cand_ind, :], centers[0:(i+1), :], squared=True)
        # Compute potentials
        p_cand = np.min(dist, axis=1)*weights[cand_ind]
        # Compute acceptance probabilities
        rand_a = random_state.random_sample(size=(chain_length))
        with nogil, cython.boundscheck(False), cython.wraparound(False), cython.cdivision(True):
            # Markov chain
            for j in range(q_cand.shape[0]):
                cand_prob = p_cand[j]/q_cand[j]
                if j == 0 or curr_prob == 0.0 or cand_prob/curr_prob > rand_a[j]:
                    # Init new chain             Metropolis-Hastings step
                    curr_ind = j
                    curr_prob = cand_prob
        centers_indices[i+1] = cand_ind[curr_ind]
        rel_row = X[centers_indices[i+1], :]
        centers[i+1, :] = rel_row.todense().flatten() if sparse else rel_row
    return centers, centers_indices
