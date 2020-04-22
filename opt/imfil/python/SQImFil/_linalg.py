import numpy

__all__ = ['kk_proj', 'f_to_vals']


#-----
def kk_proj(x, bounds):
    """
  Projection onto the feasible hyperrectangle.
 
  px = kk_proj(x, bounds)
 
  Not exciting stuff.
 
  C. T. Kelley, September 15, 2008
  This code comes with no guarantee or warranty of any kind."""

    px = numpy.minimum(bounds[:,(1,)],  x)
    return numpy.maximum(bounds[:,(0,)], px)


#-----
def f_to_vals(funs, least_squares):
    """
  Evaluate fvals = f^T f/2 when f is a vector least squares residual.
 
  fvals = f_to_vals(funs, least_squares)
 
  There is no reason you'd want to mess with this.
 
  C. T. Kelley, September 15, 2008
  This code comes with no guarantee or warranty of any kind."""

    try:
        n = funs.shape[0]
    except Exception:
        n = 1
        funs = numpy.array([funs])

    if least_squares == 1:
        fvals = zeros(n)
        for i in range(n):
            fvals[:,(i,)] = funs[:,i].dot(funs[:,(i,)])/2
    else:
        fvals = funs

    return fvals
