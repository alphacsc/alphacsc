.. alphacsc documentation master file, created by
   sphinx-quickstart on Thu Jun  1 00:35:01 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. raw:: html

	<h1>&alpha;CSC</h1>

This is a library to perform convolutional dictionary learning on neural time-series data.

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

Bug reports
===========

Use the `github issue tracker <https://github.com/alphacsc/alphacsc/issues>`_ to report bugs.

Cite
====

[1] Jas, M., Tour, L., Dupré, T., Şimşekli, U., & Gramfort, A. (2017). Learning the Morphology of
Brain Signals Using Alpha-Stable Convolutional Sparse Coding. arXiv preprint arXiv:1705.08006.