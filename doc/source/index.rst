.. sckit-quant documentation master file, created by
   sphinx-quickstart on Wed Apr 15 20:51:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
   :description: scikit-quant: scikit package for quantum computing
   :keywords: Python, C++, Quantum Computing, Optimizers

Scikit-Quant
============

Scikit-quant is a collection of optimizers tuned for usage on Noisy
Inter-mediate-Scale Quantum (NISQ) devices.
Results for several VQE and Hubbard model case studies are presented in this
`arxiv paper`_ (final paper was presented at IEEE's QCE'20).
This is the manual for the software used.


.. only: not latex

   Contents:

.. toctree::
   :maxdepth: 1

   changelog
   license

.. toctree::
   :caption: Getting Started
   :maxdepth: 1

   installation
   starting
   bugs

.. toctree::
   :caption: Optimizers
   :maxdepth: 1

   imfil
   snobfit
   nomad
   bobyqa

.. toctree::
   :caption: Interoperability
   :maxdepth: 1

   qiskit
   scipy

.. toctree::
   :caption: Developers
   :maxdepth: 1

   repositories
   testing


Bugs and feedback
-----------------

Please report bugs or requests for improvement on the `issue tracker`_.


.. _`arxiv paper`: https://arxiv.org/abs/2004.03004
.. _`issue tracker`: https://github.com/scikit-quant/scikit-quant/issues
