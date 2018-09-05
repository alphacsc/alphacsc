# αcsc
[![TravisCI](https://api.travis-ci.org/multicsc/multicsc.svg?branch=master)](https://travis-ci.org/multicsc/multicsc)
[![Codecov](https://codecov.io/github/multicsc/multicsc/coverage.svg?precision=0)](https://codecov.io/gh/multicsc/multicsc)

[Multivariate Convolutional Sparse Coding for Electromagnetic Brain Signals](https://arxiv.org/pdf/1805.09654.pdf)

[Convolutional dictionary learning for noisy signals using αCSC](https://papers.nips.cc/paper/6710-learning-the-morphology-of-brain-signals-using-alpha-stable-convolutional-sparse-coding)

Dependencies
------------

These are the dependencies

* numpy
* matplotlib
* scipy
* joblib
* mne
* cython

Installation
------------

To install this package, the easiest way is using `pip`. As this package contains
multiple cython file, please run `bash install_multicsc.sh`.

Cite
----

If you use multivariateCSC code in your project, please cite::

	Dupré La Tour, T., Moreau, T., Jas, M. & Gramfort, A. (2018).
    Multivariate Convolutional Sparse Coding for Electromagnetic Brain Signals.
    Advances in Neural Information Processing Systems 31 (NIPS)


If you use alphaCSC code in your project, please cite::

	Jas, M., Dupré La Tour, T., Şimşekli, U., & Gramfort, A. (2017).
    Learning the Morphology of Brain Signals Using Alpha-Stable Convolutional
    Sparse Coding. Advances in Neural Information Processing Systems 30 (NIPS), pages 1099--1108
