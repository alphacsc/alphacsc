"""
=======================
MoTIF on simulated data
=======================
This example demonstrates MoTIF [1] on simulated data. In the
alphacsc module, we are offering all the alternatives for the users
to try. Please cite our paper [2] if you use this implementation.

[1] Jost, Philippe, et al.
    "MoTIF: An efficient algorithm for learning
    translation invariant dictionaries." Acoustics, Speech and Signal
    Processing, 2006. ICASSP 2006 Proceedings. 2006 IEEE International
    Conference on. Vol. 5. IEEE, 2006.
[2] Jas, M., Dupré La Tour, T., Şimşekli, U., & Gramfort, A. (2017).
    Learning the Morphology of Brain Signals Using Alpha-Stable Convolutional
    Sparse Coding. arXiv preprint arXiv:1705.08006.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

###############################################################################
# Let us first import the modules.

import matplotlib.pyplot as plt
from alphacsc.simulate import simulate_data
from alphacsc.motif import learn_atoms

###############################################################################
# and define the relevant parameters. Note we choose a large n_times
# to avoid overlapping atoms which MoTIF cannot handle

n_times_atom = 64  # L
n_times = 5000  # T
n_atoms = 2  # K
n_trials = 10  # N

###############################################################################
# simulate the data.

random_state_simulate = 1
X, ds_true, Z_true = simulate_data(n_trials, n_times, n_times_atom,
                                   n_atoms, random_state_simulate,
                                   constant_amplitude=True)

n_times_atom = n_times_atom * 7  # XXX: hack
n_iter = 10
max_shift = 11  # after this, the algorithm breaks

###############################################################################
# Note, how we use constant_amplitude=True since
# MoTIF cannot handle atoms of varying amplitudes. Check out our examples
# on :ref:`vanilla CSC <sphx_glr_auto_examples_plot_simulate_csc.py>` and
# :ref:`alphaCSC <sphx_glr_auto_examples_plot_simulate_alphacsc.py>` to learn
# how to deal with such cases.
# Finally, let us estimate the atoms.

ds_hat = learn_atoms(X, n_atoms, n_times_atom, n_iter=n_iter,
                     max_shift=max_shift, random_state=42)
plt.plot(ds_hat.T)

###############################################################################
# Compare this to the original atoms
plt.figure()
plt.plot(ds_true.T)
plt.show()
