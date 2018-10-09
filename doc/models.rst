Model descriptions
==================

AlphaCSC is a library to perform shift-invariant `sparse dictionary learning
<https://en.wikipedia.org/wiki/Sparse_dictionary_learning>`_, also known as
convolutional sparse coding (CSC), on time-series data.
It includes a number of different models:

1. univariate CSC
2. multivariate CSC
3. multivariate CSC with a rank-1 constraint [1]_
4. univariate CSC with an alpha-stable distribution [2]_

1. Univariate CSC
-----------------

Framed mathematically, if we are giving a signal :math:`x^n \in \mathbb{R}^T`,
we want to learn :math:`k` atoms :math:`d^k \in \mathbb{R}^{L}` and their
associated activations :math:`z_k^n \in \mathbb{R}^{T - L + 1}`. The
optimization problem boils down to minimizing an :math:`\ell_2` reconstruction
loss, which corresponds to a Gaussian noise model, with a sparsity-inducing
:math:`\ell_1` penalty term:

.. math::
	\min_{d,z} \sum_n \left(\|x^n - \sum_k d^k * z_k^n \|_2^2 + \lambda \sum_k z_k^n \right)

subject to :math:`z_k^n \ge 0` and :math:`||d_k||_2 \le 1`. The shift invariance
is encoded by the convolution operator :math:`*` which is why these methods are
called **convolutional** sparse coding.

2. Multivariate CSC
-------------------

In the multivariate case, we are giving a multivariate signal :math:`X^n \in
\mathbb{R}^{T \times P}`, where :math:`P` is the dimension of the signal, and
we want to learn :math:`k` multivariate atoms :math:`D_k \in \mathbb{R}^{L
\times P}` and their associated activations :math:`z_k^n \in \mathbb{R}^{T - L +
1}`. The objective function reads:

.. math::
	\min_{D,z} \sum_n \left(\|X^n - \sum_k D_k * z_k^n \|_2^2 + \lambda \sum_k z_k^n \right)

subject to :math:`z_k^n \ge 0` and :math:`||D_k||_2 \le 1`.

3. Multivariate CSC with a rank-1 constraint
--------------------------------------------

A variant of the multivariate CSC model considers an additional constraint on
the multivariate atoms, stating that they need to be rank-1. In other words,
each atom can be written as a product of univariate vectors :math:`D_k = u_k
v_k^\top`, where :math:`u_k^\top \in \mathbb{R}^{P}` is the pattern over
channels, and :math:`v_k^\top \in \mathbb{R}^{L}` is the pattern over time. The
objective function reads:

.. math::
	\min_{u, v, z} \sum_n \left(\|X^n - \sum_k (u_k v_k^\top) * z_k^n \|_2^2 + \lambda \sum_k z_k^n \right)

subject to :math:`z_k^n \ge 0`, :math:`||u_k||_2 \le 1`, and :math:`||v_k||_2
\le 1`. This rank-1 formulation is particularly meaningful in the case of
magnetoencephalography (MEG), due to the physical properties of electromagnetic
waves, which propagate instantaneously and add up linearly. It also brings
robustness to the atom estimates, since less parameters are estimated.

This model is described in details in the original NIPS paper [1]_.

4. Univariate CSC with an alpha-stable distribution
---------------------------------------------------

The name alphaCSC originally referred to a particular model which uses an
`alpha-stable distribution <https://en.wikipedia.org/wiki/Stable_distribution>`_
for the noise, instead of the more classical Gaussian distribution. Note that
the ``alphaCSC`` library is **not** limited to this particular model.

This model leads to a weighted formulation of univariate CSC, the weights being
used to downweight noisy portions of the data:

.. math::
	\min_{d,z} \sum_n \left( \|\sqrt{w_n} \odot (x^n - \sum_k d_k * z_k^n) \|_2^2 + \lambda \sum_k z_k^n \right)

subject to :math:`z_k^n \ge 0` and :math:`||d_k||_2 \le 1`, and  where
:math:`w_n` are the weights which are learned using an `EM algorithm
<https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>`_.

This model is described in details in the original NIPS paper [2]_.


Cite our work
-------------

If you use this code in your project, please consider citing our work:

.. [1] Dupré La Tour, T., Moreau, T., Jas, M., & Gramfort, A. (2018).
	`Multivariate Convolutional Sparse Coding for Electromagnetic Brain Signals
	<https://arxiv.org/abs/1805.09654v2>`_. Advances in Neural Information
	Processing Systems (NIPS).

.. [2] Jas, M., Dupré La Tour, T., Şimşekli, U., & Gramfort, A. (2017). `Learning
	the Morphology of Brain Signals Using Alpha-Stable Convolutional Sparse Coding
	<https://papers.nips.cc/paper/6710-learning-the-morphology-of-brain-signals-using-alpha-stable-convolutional-sparse-coding.pdf>`_.
	Advances in Neural Information Processing Systems (NIPS), pages 1099--1108.
