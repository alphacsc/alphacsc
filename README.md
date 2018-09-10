# αcsc
[![TravisCI](https://api.travis-ci.org/alphacsc/alphacsc.svg?branch=master)](https://travis-ci.org/alphacsc/alphacsc)
[![Codecov](https://codecov.io/github/alphacsc/alphacsc/coverage.svg?precision=0)](https://codecov.io/gh/alphacsc/alphacsc)

[Multivariate Convolutional Sparse Coding for Electromagnetic Brain Signals](https://arxiv.org/pdf/1805.09654.pdf)

[Convolutional dictionary learning for noisy signals using αCSC](https://papers.nips.cc/paper/6710-learning-the-morphology-of-brain-signals-using-alpha-stable-convolutional-sparse-coding)


Installation
------------

To install this package, the easiest way is using `pip`. It will install this package and its dependencies. The `setup.py` depends on `numpy` and `cython` for the installation so it is advised to install them beforehand. To install this package, please run

```bash
pip install numpy cython
pip install .
```


Usage
-----

```python
import numpy as np
import matplotlib.pyplot as plt
from alphacsc import BatchCDL, OnlineCDL

# Define the different dimensions of the problem
n_atoms = 10
n_times_atom = 50
n_channels = 5
n_trials = 10
n_times = 1000

# Generate a random set of signals
X = np.random.randn(n_trials, n_channels, n_times)

# Learn a dictionary with online algorithm and rank1 constraints. Note that
# BatchCDL learn the atoms using a batch algorithm.
cdl = OnlineCDL(n_atoms, n_times_atom, rank1=True)
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

```


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
