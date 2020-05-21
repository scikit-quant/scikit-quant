.. _scipy:


SciPy
=====

A set of interoperable methods for SciPy's optimizer package are available in
``skquant.interop.scipy``.
These methods follow the SciPy convention, allowing them to be passed to its
``minimize`` function, such that the skquant optimmizers can be used as
drop-in replacements in SciPy-based codes

Example usage:
   
.. code-block:: python

     from skquant.interop.scipy import imfil
     from scipy.optimize import minimize

     x0 = np.array([0.5, 0.5])
     bounds = np.array([[-1, 1], [-1, 1]], dtype=float)
     budget = 40

     result = minimize(your_objective, x0, method=imfil,
                       bounds=bounds, options={'budget' : budget})

The returned ``result`` is a ``scipy.optimize.OptimizeResult`` object and
follows the same conventions for all return parameters that make sense.
Available component classes are ``imfil``, ``snobfit``, and ``pybobyqa``.
