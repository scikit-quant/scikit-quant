.. _imfil:


ImFil
=====

Implicit Filtering (ImFil) is an algorithm designed for problems
with local minima caused by high-frequency, low-amplitude noise and with an
underlying  large scale structure that is easily optimized.
ImFil uses difference gradients during the search and can be considered as an
extension  of coordinate search.
In ImFil, the optimization is controlled by evaluating the objective function
at a cluster (or stencil) of points within the given bounds.
The minimum of those evaluations then drives the next cluster of points,
using first-order interpolation to estimate the derivative, and aided by
user-provided exploration directions, if any.
Convergence is reached if the "budget" for objective function evaluations is
spent, if the smallest cluster size has been reached, or if incremental
improvement drops below a preset threshold.

The initial clusters of points are almost completely determined by the
problem boundaries, making ImFil relatively insensitive to the initial
solution and allows it to easily escape from local minima.
Conversely, this means that if the initial point is known to be of high
quality, ImFil must be provided with tight bounds around this point, or it
will unnecessarily evaluate points in regions that do not contain the global
minimum.

As a practical matter, for the noisy objective functions we studied, we find
that the total number of evaluations is driven almost completely by the
requested step sizes between successive clusters, rather than finding
convergence explicitly.

We have rewritten the original ImFil MATLAB implementation in Python

Reference:
C.T. Kelley, "Implicit Filtering", 2011, ISBN: 978-1-61197-189-7

Original software available at http://ctk.math.ncsu.edu/imfil.html
