import logging
import numpy

__all__ = ['error_check', 'check_bounds', 'check_first_eval', 'isok']

logger = logging.getLogger('SKQ.ImFil')


#-----
def isok(x, bounds):
    # test whether all values of x fall within their bounds
    il = (x >= bounds[:, 0]).all()
    iu = (x <= bounds[:, 1]).all()
    return il and iu


#-----
def error_check(*varargin):
    """
  Checks dimensions and options for input errors or inconsistencies.
  This function is not intended for use outside of imfil.m. It does
  not error check itself.
 
  C. T. Kelley, Feb 19, 2010"""
 
    name = varargin[0]
    if name == 'bounds':
        x = varargin[1]
        bounds = varargin[2]
        return check_bounds(x, bounds)
    elif name == 'first_eval':
        funs = varargin[1]
        options = varargin[2]
        iflag = varargin[3]
        return check_first_eval(funs, options, iflag)

    raise ValueError('"%s" is an illegal argument to error_check.' % name)


#-----
def check_bounds(x, bounds):
    """
  Sanity check for the bounds. Complain if the bounds are not the 
  same length as x, if the bounds are not properly arranged into 
  columns, if the lower bound is not less than the upper bound,
  or if x is infeasible."""
 
    if x.shape[0] != bounds.shape[0]:
        raise ValueError('The columns [lower, upper] of the bounds array must '
                         'have the same length as x. ')

    if bounds.shape[1] != 2:
        raise ValueError('The bounds array must have two columns [lower, upper].')

    diff_vec = bounds[:,(1,)] - bounds[:,(0,)]
    m_vec = min(diff_vec)
    if m_vec <= 0:
        raise ValueError('The lower bound must be strictly greater than the upper bound.')

    if m_vec > 0:
        px = numpy.maximum(bounds[:,(0,)], numpy.minimum(x, bounds[:,(1,)]))
        nd = numpy.linalg.norm(x-px)
        if nd > 0:
            raise ValueError('The initial iterate is infeasible. '
                             'x must satisfy the bound constraints.')


#-----
def check_first_eval(funs, options, iflag):
    """
  Complain if you have a scalar-valued function + nonlinear least squares.
  Give up if you have a vector-valued function + not a least squares problem.
  Warn if the function fails to return a value at the initial iterate."""

    lsqerr = 0
    try:
        mf, nf = funs.shape
    except (AttributeError, ValueError):
        mf = nf = 1
    lsq = options.least_squares
    if mf > 1 and lsq == 0:
        logger.warning('Your function is vector-valued but the least_squares option is off.')
        logger.warning('imfil cannot run with this inconsistency.')
        lsqerr = 1

    elif mf == 1 and lsq == 1:
        logger.warning('Your function is scalar-valued but the least_squares option is on.')
        logger.warning('Are you sure this is what you want to do?')

    if iflag > 0:
        logger.warning('Your function failed to return a value at the initial iterate.')
        logger.warning('This is usually a problem. You have been warned.')

    return lsqerr
