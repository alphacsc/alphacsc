===================================================
alphaCSC: Convolution sparse coding for time-series
===================================================
|Build Status| |codecov|

This is a library to perform shift-invariant `sparse dictionary learning
<https://en.wikipedia.org/wiki/Sparse_dictionary_learning>`_, also known as
convolutional sparse coding (CSC), on time-series data.
It includes a number of different models:

1. univariate CSC
2. multivariate CSC
3. multivariate CSC with a rank-1 constraint [1]_
4. univariate CSC with an alpha-stable distribution [2]_

A mathematical descriptions of these models is available `in the documentation
<https://alphacsc.github.io/models.html>`_.

Installation
============

To install this package, the easiest way is using ``pip``. It will install this
package and its dependencies. The ``setup.py`` depends on ``numpy`` and
``cython`` for the installation so it is advised to install them beforehand. To
install this package, please run one of the two commands:

(Latest stable version)

.. code::

    pip install alphacsc

(Development version)

.. code::

	pip install git+https://github.com/alphacsc/alphacsc.git#egg=alphacsc

(Dicodile backend)

.. code::
   
   pip install numpy cython
   pip install alphacsc[dicodile]

To use dicodile backend, do not forget to set ``MPI_HOSTFILE`` environment
variable.


If you do not have admin privileges on the computer, use the ``--user`` flag
with ``pip``. To upgrade, use the ``--upgrade`` flag provided by ``pip``.

To check if everything worked fine, you can run:

.. code::

	python -c 'import alphacsc'

and it should not give any error messages.


Quickstart
==========

Here is an example to present briefly the API:

.. code:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from alphacsc import BatchCDL

    # Define the different dimensions of the problem
    n_atoms = 10
    n_times_atom = 50
    n_channels = 5
    n_trials = 10
    n_times = 1000

    # Generate a random set of signals
    X = np.random.randn(n_trials, n_channels, n_times)

    # Learn a dictionary with batch algorithm and rank1 constraints.
    cdl = BatchCDL(n_atoms, n_times_atom, rank1=True)
    cdl.fit(X)

    # Display the learned atoms
    fig, axes = plt.subplots(n_atoms, 2, num="Dictionary")
    for k in range(n_atoms):
        axes[k, 0].plot(cdl.u_hat_[k])
        axes[k, 1].plot(cdl.v_hat_[k])

    axes[0, 0].set_title("Spatial map")
    axes[0, 1].set_title("Temporal map")
    for ax in axes.ravel():
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.show()

Dicodile backend
================

AlphaCSC can use a `dicodile <https://github.com/tomMoral/dicodile>`_-based backend to perform sparse encoding in parallel.

To install dicodile, run ``pip install alphacsc[dicodile]``.

Known OpenMPI issues
--------------------

When self-installing OpenMPI (for instance to run `dicodile` on a single machine, or for continuous integration), running the `dicodile` solver might end up causing a deadlock (no output for a long time). It is often due to communication issue between the workers. This issue can often be solved by disabling Docker-related virtual NICs, for instance by running ``export OMPI_MCA_btl_tcp_if_exclude="docker0"``.

Bug reports
===========

Use the `github issue tracker <https://github.com/alphacsc/alphacsc/issues>`_ to report bugs.

Cite our work
=============

If you use this code in your project, please consider citing our work:

.. [1] Dupré La Tour, T., Moreau, T., Jas, M., & Gramfort, A. (2018).
	`Multivariate Convolutional Sparse Coding for Electromagnetic Brain Signals
	<https://arxiv.org/abs/1805.09654v2>`_. Advances in Neural Information
	Processing Systems (NIPS).

.. [2] Jas, M., Dupré La Tour, T., Şimşekli, U., & Gramfort, A. (2017). `Learning
	the Morphology of Brain Signals Using Alpha-Stable Convolutional Sparse Coding
	<https://papers.nips.cc/paper/6710-learning-the-morphology-of-brain-signals-using-alpha-stable-convolutional-sparse-coding.pdf>`_.
	Advances in Neural Information Processing Systems (NIPS), pages 1099--1108.

.. |Build Status| image:: https://github.com/alphacsc/alphacsc/workflows/unittests/badge.svg
.. |codecov| image:: https://codecov.io/gh/alphacsc/alphacsc/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/alphacsc/alphacsc
