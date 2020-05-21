.. _starting:

Trying it out
=============

This is a basic guide to using the optimizers mainly intended to test whether
your installation works.
If you are already familiar to using optimizers within a quantum programming
framework, you may be better served using the component interfaces, such as
the ones to :doc:`Qiskit <qiskit>` and :doc:`SciPy <scipy>`.

First, you need to have some objective function to optimize.
All the optimizers are *minimizers* and expect to do simple "less than"
comparisons on the result.
Thus, if instead you need to maximize the result, simply add a minus sign.
The objective function is expected to accept an evaluation point in the form
of a numpy array of floating point values, or a list of such evaluation
points to allow evaluation in parallel.

Example of an objective function:

.. code-block:: python

    import numpy as np

    # some interesting objective function to minimize
    def objective_function(x):
        fv = np.inner(x, x)
        fv *= 1 + 0.1*np.sin(10*(x[0]+x[1]))
        return np.random.normal(fv, 0.01)

All optimizers provided require bounds.
This is not true for optimizers in general, but is of such great benefit when
dealing with noisy objective functions that it is pretty much a requirement.
In most cases, the better the bounds, the faster the optimizer will run and
the higher the quality of the result.
For difficult problems, it may be necessary to refine bounds while switching
optimizers to solve.
Not all optimizers are equally sensitive to bounds.

.. code-block:: python 

   # create a numpy array of bounds, one (low, high) for each parameter
   bounds = np.array([[-np.pi, np.pi], [-np.pi, np.pi]], dtype=float)

Likewise, consider whether a good initial estimate can be provided, and if
yes, it is often worthwhile to spend some (classical) computational resources
to obtain a high quality initial estimate.
Not every optimizer benefits equally of a good initial estimate, but most do,
especially when combined with tight bounds.
If no initial estimate is provided, a random point is used within the given
bounds.

.. code-block:: python

   # initial values for all parameters
   x0 = np.array([0.5, 0.5])

The objective function is considered expensive to calculate (running a
circuit many times on the QPU).
It is therefore better to consider a *budget* (number of allowed
evaluations), rather than rely on convergence criteria, especially since
tight tolerances can not be met in the case of large noise.
The budget is an upper limit.
If convergence happens earlier, the minimizer will stop.

.. code-block:: python

   # budget (number of calls, assuming 1 count per call)
   budget = 100

Finally, import and run the minimizer.
The result object will contain the optimal parameters (``result.optpar``) and
optimal value (``result.optval``).
The history object contains the full call history.

.. code-block:: python

   from skquant.opt import minimize

   # method can be ImFil, SnobFit, Orbit, or Bobyqa
   result, history = \
       minimize(objective_function, x0, bounds, budget, method='imfil')
