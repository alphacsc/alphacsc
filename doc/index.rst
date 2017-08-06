.. alphacsc documentation master file, created by
   sphinx-quickstart on Thu Jun  1 00:35:01 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. raw:: html

	<h1>&alpha;CSC</h1>

This is a library to perform shift-invariant
`dictionary learning <https://en.wikipedia.org/wiki/Sparse_dictionary_learning>`_ on neural time-series data.
Framed mathematically, if we are giving a signal :math:`x_n \in \mathbb{R}^T`, we want to learn :math:`k`
atoms :math:`d^k \in \mathbb{R}^{L}` and their associated activations :math:`z^k_n \in \mathbb{R}^{T - L + 1}`. The optimization
problem boils down to minimizing an :math:`\ell_2` reconstruction loss with an :math:`\ell_1` penalty term, i.e.,

.. math::
	\min_{d,z} \sum_n (\|x_n - \sum_k d^k * z^k_n \|_2^2 + \lambda \sum_k z_n^k)

subject to :math:`z_k >= 0`. The shift invariance is encoded by the convolution operator :math:`*` which is
why these methods are called "convolutional sparse coding".

The alpha in alphaCSC [1] refers to the fact that we assume an `alpha-stable
distribution <https://en.wikipedia.org/wiki/Stable_distribution>`_ for the noise.
This leads to a weighted formulation of `vanilla' CSC, the weights being used to downweight
noisy portions of the data. Or in other words:

.. math::
	\min_{d,z} \sum_n (\|\sqrt{w_n} \odot (x_n - \sum_k d^k * z^k_n) \|_2^2 + \lambda \sum_k z_k)

where :math:`w_n` are the weights which are learned using an `EM algorithm <https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>`_.

Installation
============

We recommend the `Anaconda Python distribution <https://www.continuum.io/downloads>`_. To install ``alphacsc``, you first need to install its dependencies::

	$ pip install numpy matplotlib scipy

Then install alphacsc::

	$ pip install git+https://github.com/alphacsc/alphacsc.git#egg=alphacsc

If you do not have admin privileges on the computer, use the ``--user`` flag
with `pip`. To upgrade, use the ``--upgrade`` flag provided by `pip`.

To check if everything worked fine, you can do::

	$ python -c 'import alphacsc'

and it should not give any error messages.

Quickstart
==========

All you need is a numpy array ``X`` of shape `n_trials` x `n_times`. You need to specify the
length of atoms `n_times_atom` and the number of atoms `n_atoms` you would like to estimate.
The penalty parameter `reg` controls the sparsity in the activations estimated.

.. code:: python

>>> from alphacsc import learn_d_z
>>> n_atoms, n_times_atom, n_iter = 2, 64, 60
>>> pobj, times, d_hat, Z_hat = learn_d_z(X, n_atoms, n_times_atom, reg=reg, n_iter=n_iter)  # doctest: +SKIP

Bug reports
===========

Use the `github issue tracker <https://github.com/alphacsc/alphacsc/issues>`_ to report bugs.

Cite
====

[1] Jas, M., Dupré La Tour, T., Şimşekli, U., & Gramfort, A. (2017). `Learning the Morphology of
Brain Signals Using Alpha-Stable Convolutional Sparse Coding <https://arxiv.org/pdf/1705.08006>`_.
arXiv preprint arXiv:1705.08006.
