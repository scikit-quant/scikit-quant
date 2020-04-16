scikit-quant
============

scikit-quant is an aggregator package to improve interoperability between
quantum computing software packages.
Our first focus in on classical optimizers, making the state-of-the art from
the Applied Math community available in Python for use in quantum computing.

Full documentation: https://scikit-quant.readthedocs.io/
Website: http://scikit-quant.org


Installation
------------

   pip install sckit-quant


Usage
-----

Basic example::

   # create a numpy array of bounds, one (low, high) for each parameter
   bounds = np.array([[-1, 1], [-1, 1]], dtype=float)

   # budget (number of calls, assuming 1 count per call)
   budget = 40

   # initial values for all parameters
   x0 = np.array([0.5, 0.5])

   # method can be ImFil, SnobFit, Orbit, or Bobyqa
   result, history = \
       minimize(objective_function, x0, bounds, budget, method='imfil')
