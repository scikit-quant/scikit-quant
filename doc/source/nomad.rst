.. _nomad:


NOMAD
=====

NOMAD, or"Nonlinear Optimization by Mesh Adaptive Direct Search (MADS)" is a
C++ implementation of the MADS algorithm.
MADS searches the parameter space by iteratively generating a new sample
point from a  mesh that is adaptively adjusted based on the progress
of the search. If the newly selected sample point does not improve the current
best point, the mesh is refined. NOMAD uses   two  steps ({\em search} and
{\em poll}) alternately until some preset stopping criterion (such
as minimum mesh size, maximum number of failed consecutive trials, or maximum
number of steps) is met.
The search step can return any point on the current mesh, and therefore offers no
convergence guarantees. % if the objective function results are noisy.
If the search step fails to find an improved solution, the poll step is used to
explore  the neighborhood of the current best
solution. The poll step is central to the convergence analysis of NOMAD, and
therefore any hyperparameter optimization or other tuning to make progress should
focus on the poll step.
Options include: poll direction type (local model, random, uniform angles,
etc.), poll size, and number of polling points.

The use of meshes means that the number of evaluations needed scales at least
geometrically with the number of parameters to be optimized.
It is therefore important to restrict  the search space as much as possible
using bounds and, if the science of the problem so indicates, give preference
to polling directions of the more important parameters.

We incorporate the published open-source NOMAD code through a modified Python
interface.

Reference: C. Audet and J. Dennis, Jr. Mesh adaptive direct search algorithms for
constrained optimization. SIAM Journal on Optimization, 17(1):188–217, 2006.

Software available at: https://www.gerad.ca/nomad/
