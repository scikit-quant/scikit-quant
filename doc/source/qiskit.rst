.. _qiskit:


Qiskit
======

A set of interoperable components for Qiskit Aqua's optimizer package are
available in ``skquant.interop.qiskit``.
These classes derive from Qiskit's ``Optimizer`` class and implement the
same interface, such that the skquant optimmizers can be used as drop-in
replacements in Qiskit-based codes.

.. caution::

     The optimizer classes in Qiskit's ``Optimizer`` package do not follow
     proper conventions themselves.
     In writing the interop component classes, an attempt was made to stick
     to the most prevalent conventions present as of ``Aqua`` version 0.7.1.

Example usage:

.. code-block:: python

     from skquant.interop.qiskit import SnobFit

     x0 = np.array([0.5, 0.5])
     bounds = np.array([[-1, 1], [-1, 1]], dtype=float)

     optimizer = SnobFit(maxfun=40, maxmp=len(x0)+6)

     ret = optimizer.optimize(num_vars=len(x0),
                              objective_function=your_objective,
                              variable_bounds=bounds,
                              initial_point=x0)

Available component classes are ``ImFil``, ``SnobFit``, and ``PyBobyqa``.
