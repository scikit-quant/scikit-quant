from __future__ import print_function

import logging
import skquant.opt as skqopt
import scipy.optimize as spopt

__all__ = [
    'imfil',
    'snobfit',
    'pybobyqa',
    'bobyqa',
    ]

log = logging.getLogger('SKQ')


def _res2scipy(result, history):
    ret = spopt.OptimizeResult()

  # the following values need refinement
    ret.success = True
    ret.status  = 0
    ret.message = 'completed'

  # the following are proper
    ret.x    = result.optpar.copy()
    ret.fun  = result.optval
    ret.nfev = len(history)

    return ret

def _split_options(options):
    try:
        budget = options.pop('budget')
        bounds = options.pop('bounds')
        for name in ['args', 'jac', 'hess', 'hessp', 'constraints', 'callback']:
            options.pop(name, None)
        return budget, bounds, options
    except KeyError:
        pass

    raise RuntimeError('missing bounds or budget in options')


def imfil(fun, x0, *args, **options):
    """
    Implicit Filtering

    Algorithm designed for problems with local minima caused by high-frequency,
    low-amplitude noise, with an underlying large scale structure. This uses
    the SQImFil Python rewrite.

    Reference:
      C.T. Kelley, "Implicit Filtering", 2011, ISBN: 978-1-61197-189-7

    Original MATLAB code available at ctk.math.ncsu.edu/imfil.html
    """

    budget, bounds, options = _split_options(options)

    result, history = skqopt.minimize(fun, x0, bounds=bounds, budget=budget, \
                                      method='imfil', options=options)

    return _res2scipy(result, history)


def snobfit(fun, x0, *args, **options):
    """
    Stable Noisy Optimization by Branch and FIT

    SnobFit is specifically developed for optimization problems with noisy and
    expensive to compute objective functions. This implementation uses the
    SQSnobFit Python rewrite.

    Reference:
      W. Huyer and A. Neumaier, “Snobfit - Stable Noisy Optimization by Branch
      and Fit”, ACM Trans. Math. Software 35 (2008), Article 9.

    Original MATLAB code available at www.mat.univie.ac.at/~neum/software/snobfit
    """

    budget, bounds, options = _split_options(options)

    result, history = skqopt.minimize(fun, x0, bounds=bounds, budget=budget, \
                                      method='snobfit', options=options)

    return _res2scipy(result, history)


def pybobyqa(fun, x0, *args, **options):
    """
    Bound Optimization BY Quadratic Approximation

    Trust region method that builds a quadratic approximation in each iteration
    based on a set of automatically chosen and adjusted interpolation points.

    Reference:
      Coralia Cartis, et. al., “Improving the Flexibility and Robustness of
      Model-Based Derivative-Free Optimization Solvers”, technical report,
      University of Oxford, (2018).

    Code available at github.com/numericalalgorithmsgroup/pybobyqa/
    """

    budget, bounds, options = _split_options(options)

    result, history = skqopt.minimize(fun, x0, bounds=bounds, budget=budget, \
                                      method='bobyqa', options=options)

    return _res2scipy(result, history)

bobyqa = pybobyqa
