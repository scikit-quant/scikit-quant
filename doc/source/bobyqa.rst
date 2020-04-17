.. _bobyqa:


PyBobyqa
========

BOBYQA (Bound Optimization BY Quadratic Approximation) has been
designed to minimize bound constrained black-box optimization problems.
BOBYQA employs a trust region method and builds a quadratic approximation in
each iteration that is based on a set of automatically chosen and adjusted
interpolation points.
New sample points are iteratively created by either a "trust region" or an
"alternative iterations" step.
In both methods, a vector (step) is chosen and added to the current iterate to
obtain the new point.
In the trust region step, the vector is determined such that it minimizes the
quadratic model around the current iterate and lies within the trust region.
It is also ensured that the new point (the sum of the vector and the current
iterate) lies within the parameter upper and lower bounds.
BOBYQA uses the alternative iteration step whenever the norm of the vector is
too small, and would therefore reduce the accuracy of the quadratic model.
In that case, the vector is chosen such that good linear independence of the
interpolation points is obtained.
The current best point is updated with the new point if the new function value
is better than the current best function value.
Note that there are some restrictions for the choice of  the initial point due
to the requirements for constructing the quadratic model.
BOBYQA may thus adjust the initial automatically if needed.

Although it is not intuitively obvious that BOBYQA would work well on noisy
problems, we find that it performs well in practice if the initial parameters
are quite close to optimal and the minimum and maximum sizes of the trust
region are properly set.
This is rather straightforward to do for the specific case of VQE, where a
good initial guess can be obtained relatively cheaply from classical simulation.
For Hubbard model problems, which have many (shallow) local minima, BOBYQA
does not perform nearly as well.

We use the existing PyBobyqa implementation directly from PyPI.

Reference: Coralia Cartis, et. al., "Improving the Flexibility and Robustness of
Model-Based Derivative-Free Optimization Solvers", technical report,
University of Oxford, (2018).

Software available at http://github.com/numericalalgorithmsgroup/pybobyqa/
