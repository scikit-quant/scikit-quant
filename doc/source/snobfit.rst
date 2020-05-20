.. _snobfit:


SnobFit
=======

Stable Noisy Optimization by Branch and FIT (SnobFit) is an
optimizer developed specifically for optimization problems with noisy and
expensive to compute objective functions.
SnobFit iteratively selects a set of new evaluation points such that a balance
between global and local search is achieved, and thus the algorithm can escape
from local optima.
Each call to SnobFit requires the input of a set of evaluation points and
their corresponding function values and SnobFit returns a new set of points to
be evaluated, which is used as input for the next call of SnobFit.
Therefore, in a single optimization, SnobFit is called several times.
The initial set of points is provided by the user and should contain as many
expertly chosen points as possible (if too few are given, the choice is a
uniformly random set of points, and thus providing good bounds becomes important).
In addition to these points, the user can also specify the uncertainties
associated with each function value.
We have not exploited this feature in our test cases, because although we know
the actual noise values from the simulation, properly estimating whole-circuit
systematic errors from real hardware is an open problem.

As the name implies, SnobFit uses a branching algorithm that recursively
subdivides the search space into smaller subregions from which evaluation
points are chosen.
In order to search locally, SnobFit builds a local quadratic model around the
current best point and minimizes it to select one new evaluation point.
Other local search points are chosen as approximate minimizers within a trust
region defined by safeguarded nearest neighbors.
Finally, SnobFit also generates points in unexplored regions of the parameter
space and this represents the more global search aspect.

We have rewritten the original SnobFit MATLAB implementation in Python

Reference: W. Huyer and A. Neumaier, "Snobfit - Stable Noisy Optimization by
Branch and Fit", ACM Trans. Math. Software 35 (2008), Article 9.

Original software available at http://www.mat.univie.ac.at/~neum/software/snobfit
