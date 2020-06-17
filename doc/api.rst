.. _api_documentation:

=================
API Documentation
=================

.. currentmodule:: alphacsc

Transformers
============

.. autosummary::
   :toctree: generated/

   BatchCDL
   GreedyCDL
   OnlineCDL

Functions
=========

Functions to learn atoms (d) and activations (z) from the signal

.. autosummary::
   :toctree: generated/

   learn_d_z
   learn_d_z_weighted

Utility functions (:py:mod:`alphacsc.utils`):

.. currentmodule:: alphacsc.utils

.. autosummary::
   :toctree: generated/

   check_univariate_signal
   check_multivariate_signal
   split_signal
