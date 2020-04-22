from ._linalg import kk_proj, f_to_vals
from ._history import CompleteHistory, single_point_hist_update
import logging
import numpy
import operator

__all__ = ['gauss_newton', 'qn_update']

logger = logging.getLogger('SKQ.ImFil')


#-----
def gauss_newton(f, x, fun, jac, xc,  gc, iteration_data, hessold):
    """
  Compute a damped Gauss-Newton step.
 
  xp, fvalp, funs, qfct, iarm, hess = \
          gauss_newton(f, x, fun, jac, xc, gc, iteration_data, hessold)
 
  You may elect to modify this routine if you don't want to spend the
  effort computing the Gauss-Newton model Hessian. The current version
  of imfil.m does not use it.
 
  Inputs: 
          f = objective function; f returns an residual vector.
          x = current point.
        fun = current (vector) residual at x.
        jac = stencil Jacobian at x.
         xc = previous point; xc is not used in this function, but
              that may change if we elect to put a quasi-Newton 
              method in here.
         gc = stencil gradient at xc.
  iteration_data = internal structure with many goodies inside; rtfm
    hessold = Gauss-Newton model Hessian at xc. Dummy argument
              waiting for quasi-Newton or Levenberg-Marquardt.
 
  Output:
       hess = Gauss-Newton model Hessian at x; no update done in here.
              It's in the argument lists only to enable a quasi-Newton
              update or a Levenberg-Marquardt iteration.
         xp = new point.
      fvalp = least squares error at xp = funs'*funs/2;
       funs = vector residual at xp.
       qfct = cost in function evaluations.
       iarm = number of step size reductions.
  diff_hist = history data for the Gauss-Newton loop
      nfail = 0 if the line search succeeds, 1 if it fails
         
  C. T. Kelley, January 10, 2011
  This code comes with no guarantee or warranty of any kind.
 
  Harvest what you need from iteration_data."""

    obounds   = iteration_data.obounds
    options   = iteration_data.options
    core_data = iteration_data.core_data
    h         = iteration_data.h

    # Compute stencil gradient (sgrad) and least squares error (fval)
    # at current point.
    funs = fun
    fval = fun.dot(fun)/2.
    sgrad = jac.dot(fun)

    hess = jac.dot(jac)  # Return the normal equations model Hessian for now

    n = len(x)

    # Get the epsilon-active indices and encode them in a diagonal matrix.
    epsb = 1.E-6;
    alist = numpy.zeros(len(x)).T
    for i in range(len(x)):
        alist[i] = (x[i] > obounds[i,0]+epsb) and (x[i] < obounds[i,1]-epsb)
    pr = numpy.diagflat(alist)

    # Compute the search direction with a QR factorization.
    rjac = jac.dot(pr)
    rq, rr = numpy.linalg.qr(rjac)
    sdir1 = (numpy.eye(n)-pr)*sgrad
    sdir2 = rq.dot(fun)
    sdir2 = numpy.linalg.lstsq(rr, sdir2.T, rcond=None)[0]
    sdir = sdir1+sdir2

    # Bound constrained line search
    qfct, xp, fvalp, iarm, fres, diff_hist, nfail = \
        armijo_explore(f, sdir, fval, x, h, core_data, obounds)

    # If the line search fails nothing changes.
    if len(fres) > 0:
       funs = fres

    return (xp, fvalp, funs, qfct, iarm, diff_hist, nfail, hess)


#-----
def qn_update(f, x, fval, sgrad, xc, gc, iteration_data, hessold):
    """
  hess, xp, fvalp, funs, qfct, iarm, diff_hist = \
          imfil_qn_update(f, x, fval, sgrad, xc, gc, iteration_data, hessold)
  
  Quasi-Newton update of point and Hessian. This function is never called
  for least squares problems.
 
  Input: 
         f = objective function.
         x = current point.
      fval = f(x). 
     sgrad = stencil gradient at current point x.
        xc = previous point.
        gc = stencil gradient at previous point xc.
  iteration_data = imfil internal structure
   hessold = previous model Hessian
 
  Output:
      hess = Quasi-Newton update Hessian at x.
        xp = new point.
     fvalp = f(xp)
     funs  = fvalp; Makes calling sequence compatible with Gauss-Newton
      qfct = cost in function evaluations.
      iarm = number of step size reductions.
  diff_hist= history data for the quasi-Newton loop
      nfail= 0 if line search succeeds, 1 if it fails
 
  The steps are (1) update Hessian to obtain hess(current Hessian),
                (2) use hess, gc, and the bounds to compute the new
                    search direction,
                (3) do a line search on that direction,
 
  C. T. Kelley, January 23, 2011
  This code comes with no guarantee or warranty of any kind."""

    core_data = iteration_data.core_data

    # obounds are the scaled 0-1 bounds. I use this field because my quasi-Newton
    # codes were written for general bounds and I do not want to invade them
    # any more than necessary.
    obounds = iteration_data.obounds
    h       = iteration_data.h

    # itc is the inner iteration counter. I am not updating the model Hessian
    # for the first inner iteration. The reason for this is that I need a 
    # couple gradients at each scale to make the quasi-Newton formula make sense.
    itc     = iteration_data.itc
    options = core_data.options
    quasi   = options.quasi

    nx = len(x)
    hess = hessold.copy()

    # Update the model Hessian for all but the initial inner iteration.
    if itc > 1:
        # Get the epsilon-inactive indices.
        epsb=1.E-6;
        alist = numpy.zeros(len(x)).T
        for i in range(len(x)):
            alist[i] = (x[i] > obounds[i,0]+epsb) and (x[i] < obounds[i,1]-epsb)

        if quasi == 1:        # BFGS
            hess = bfupdate(x, xc, sgrad, gc, hessold, alist)
        elif quasi == 2:      # SR1
            hess = sr1up(x, xc, sgrad, gc, hessold, alist)
        else:                 # nothing
            hess = numpy.eye(nx, nx)

    # Search direction
    sdir = numpy.linalg.lstsq(hess, sgrad, rcond=None)[0]

    # Bound constrained line search
    #
    qfct, xp, fvalp, iarm, fres, diff_hist, nfail = \
          armijo_explore(f, sdir, fval, x, h, core_data, obounds)
    funs = fvalp

    return (xp, fvalp, funs, qfct, iarm, diff_hist, nfail, hess)


#-----
def bfupdate(x, xc, sgrad, gc, hess, alist):
    """BFGS update of reduced Hessian, nothing fancy."""

    n = len(x)
    pr = numpy.diagflat(alist)
    y = sgrad-gc; s = x-xc; z = hess.dot(s)

    # Turn y into y#.
    y = pr.dot(y)
    if y.T.dot(s) > 0:
        hess = pr.dot(hess).dot(pr) + (y.dot(y.T)/(y.T.dot(s))) - pr.dot((z.dot(z.T)/(s.T.dot(z)))).dot(pr)
        hess = numpy.eye(n) - pr + hess

    if numpy.linalg.cond(hess) > 1.E6:
        hess = numpy.eye(n)

    return hess


#-----
def sr1up(x, xc, sgrad, gc, hess, alist):
    """SR1 update of reduced Hessian."""

    n = len(x)
    pr = numpy.diagflat(alist)
    y = sgrad-gc; s=x-xc
    z = y - hess*s

    # Turn y into y#.
    y = pr*y; z = pr*z
    if z.T*s != 0:
        ptst = z.T*(hess*z)+(z.T*z)*(z.T*z)/(z.T*s)
        if ptst > 0:
            hess = pr*hess*pr + (z*z.T)/(z.T*s);
            hess = numpy.eye(n) - pr + hess

    if numpy.linalg.cond(hess) > 1.E6:
        hess = numpy.eye(n)

    return hess


#-----
def armijo_explore(f, sdir, fold, xc, h, core_data, obounds):
    """
  Line search for imfil.m.
 
  C. T. Kelley, September 15, 2008
 
  This code comes with no guarantee or warranty of any kind.
 
  fct, x, fval, iarm, fres, diff_hist, nfail = \
        armijo_explore(f, sdir, fold, xc, h, core_data, obounds)
 
  This is an internal function, which you are NOT TO HACK! Since
  you may hack it anyway, I will tell you want is going on. If you
  break something, may Alberich's curse be upon you!
 
  Inputs:
          f = objective function.
       sdir = quasi-Newton search direction.
       fold = current function value.
   maxitarm = limit on number of step size reductions.
       beta = stepsize reduction factor.
          h = current scale.
  core_data = structure with the options + fun_data
              The options are documented in optset.m.
              fun_data is private to imfil.m and is used in the internal
              scaling for the function evaluation.

    obounds = Nx2 array of scaled bound constraints from imfil_core.

 Output:
        fct = cost of line search in function evaluations.
          x = new point.
       fval = f(x).
       iarm = number of step length reductions.  
       fres = residual for nonlinear least squares problems.
  diff_hist = history data for this iteration.
      nfail = 0 if the line search succeeds, 1 if it fails.

  Read the options array to get the ones we care about in the line search."""

    options = core_data.options

    if not options.parallel:
        fct, x, fval, iarm, aflag, fres, diff_hist, nfail = \
            serial_armijo(f, sdir, fold, xc, h, obounds, core_data)
    else:
        fct, x, fval, iarm, aflag, fres, diff_hist, nfail = \
            parallel_armijo(f, sdir, fold, xc, h, obounds, core_data)

    if iarm == options.maxitarm and aflag == 1 and options.verbose == 1:
        logger.warning('line search failure %d %s', iarm, str(h))

    return (fct, x, fval, iarm, fres, diff_hist, nfail)


#-----
def parallel_armijo(f, sdir, fold, xc, h, obounds, core_data):
    """
  Parallel line search for imfil.m.
  Uses extra processors to explore larger stencils.
 
  fct, x, fval, iarm, aflag, fres = \
     parallel_armijo(f, sdir, fold, xc, h, obounds, core_data)
 
  Before you modify this (to use a trust region method, say), consider
  writing your own with this as a template. I'll make this very easy
  in a later version of the code. 
 
  Inputs:
           f = objective function or residual vector, depending
               on whether this is a least squares problem or not.
        sdir = search direction.
        fold = current value of the objective.
        xc   = current point.
    maxitarm = limit on number of step size reductions.
        beta = reduction factor of step size. Currently beta = 1/2.
           h = current scale.
     obounds = Nx2 array of scaled bound constraints from imfil_core.
 
   core_data = structure with the options + fun_data
               The options are documented in optset.m.
               fun_data is private to imfil.m and is used in the internal
               scaling for the function evaluation.
  least_squares = least squares flag.
 
  Output:
         fct = cost in function evaluations.
           x = new point.
        fval = objective function value at new point = fres'*fres/2
               in the nonlinear least squares case.
       iarm  = number of step size reductions.
       aflag = 0 if the search finds a new point, = 1 if not.
       fres  = residual for nonlinear least squares problems
   diff_hist = history structure for this iteration
       nfail = 0 if the line search succeeds, 1 if it fails.
 
  C. T. Kelley, July 21, 2009
  This code comes with no guarantee or warranty of any kind."""

    options = core_data.options
    least_squares = options.least_squares
    beta          = options.armijo_reduction
    maxitarm      = options.maxitarm

    #Initialize the line search.
    lbda = 1
    n = len(xc)
    fct = 0
    aflag = 1
    dd = sdir
    fres = []
    frest = fres
    diff_hist = CompleteHistory()

    if options.limit_quasi_newton:
        # If the quasi-Newton step is much longer than the scale, shrink it.
        #
        # I am not convinced that this is the right thing to do, but it
        # really helps most of the time.
        #
        smax = 10*min(h, 1)
        if numpy.linalg.norm(dd) > smax:
            dd = smax*dd/numpy.linalg.norm(dd)

    x = xc
    fval = fold

    # Evaluate all steplength choices at once.
    number_steps = maxitarm+1
    ddm = numpy.zeros((n, number_steps))
    for i in range(number_steps):
        ddm[:,(i,)] = xc-lbda*dd
        lbda = beta*lbda
        ddm[:,(i,)] = kk_proj(ddm[:,(i,)], obounds)

    fta, iflaga, ictra, tol = f(ddm, h, core_data)

    # Sort the points in failed and good into the history structure.
    for i in range(len(iflaga)):
        iflag = iflaga[i]
        if iflag:
            diff_hist.failed_points.append(ddm[:,i])
        else:
            diff_hist.good_points.append(ddm[:,i])
            if least_squares == 1:
                diff_hist.good_values.append(fta[:,i])
            else:
                diff_hist.good_values.append(fta[i])

    if least_squares:
        frest = fta

    fta = f_to_vals(fta, least_squares)
    fct += sum(ictra)
    ilose = (iflaga==1)
    fta[ilose] = fold+abs(fold)*.0001+1.E-12

    # Traditional duplicates a serial Armijo search. This will take
    # the longest possible step and is sometimes the right thing to do.
    #
    # Non-traditional will take the best point from all the
    # ones sampled in the search direction if the full step fails.
    #
    # I am still playing with this, and will let you chose in the options
    # command pretty soon. 
    traditional = 0
    if traditional == 0:
        it, ft = min(enumerate(fta), key=operator.itemgetter(1))
        xt = ddm[:, it]
        iarm = maxitarm
        if ft < fval:
            aflag = 0
            fval = ft
            if least_squares:
                fres = frest[:,it]
            x = xt
            iarm = min(it, maxitarm-1)
        else:
            iarm -= 1

            while iarm < maxitarm and aflag == 1:
                ft = fta[iarm+2]
                xt = ddm[:,iarm+2]
                if ft < fval and aflag == 1:
                    aflag = 0
                    if least_squares:
                        fres = frest[:,iarm+2]
                    fval = ft
                    x = xt
                iarm += 1

    if iarm == maxitarm and aflag == 1 and options.verbose == 1:
        logger.warning('line search failure %d %s', iarm, str(h))

    if len(x.shape) == 1:
        x = x.reshape(len(x), 1)
    nfail = aflag
    return (fct, x, fval, iarm, aflag, fres, diff_hist, nfail)


#-----
def serial_armijo(f, sdir, fold, xc, h, obounds, core_data):
    """
  Serial line search for imfil.m.
 
  fct, x, fval, iarm, aflag, fres = \
          serial_armijo(f, sdir, fold, xc, h, obounds, core_data)
 
  This is a plain vanilla serial line search. Nothing to see here;
  move along.
 
  Before you modify this (to use a trust region method, say), consider
  writing your own with this as a template. I'll make this very easy
  in a later version of the code.
 
  Inputs:
            f = objective function or residual vector, depending
                on whether this is a least squares problem or not.
         sdir = search direction.
         fold = current value of the objective.
           xc = current point.
            h = current scale.
      obounds = Nx2 array of scaled bound constraints from imfil_core.
 
    core_data = structure with the options + fun_data
                The options are documented in optset.m.
                fun_data is private to imfil.m and is used in the internal
                scaling for the function evaluation.
 
  Output:
         fct = cost in function evaluations.
           x = new point.
        fval = objective function value at new point = fres'*fres/2
               in the nonlinear least squares case.
       iarm  = number of step size reductions.
       aflag = 0 if the search finds a new point, = 1 if not.
       fres  = residual for nonlinear least squares problems
   diff_hist = history structure for this iteration
       nfail = 0 if the line search succeeds, 1 if it fails.

  C. T. Kelley, September 15, 2008
  This code comes with no guarantee or warranty of any kind."""

    options = core_data.options;
    least_squares = options.least_squares
    beta          = options.armijo_reduction
    maxitarm      = options.maxitarm

    # Initialize the line search.
    lbda = 1
    n = len(xc)
    iarm = -1
    fct = 0
    aflag = 1
    dd = sdir
    fres = []
    frest = []
    diff_hist = CompleteHistory()

    if options.limit_quasi_newton:
        # If the quasi-Newton step is much longer than the scale, shrink it.
        #
        # I am not convinced that this is the right thing to do, but it
        # really helps most of the time.
        smax = 10*min(h, 1)
        if numpy.linalg.norm(dd) > smax:
            dd = smax*dd/numpy.linalg.norm(dd)

    x = xc
    fval = fold
    while iarm < maxitarm and aflag == 1:
        d = -lbda*dd
        xt = x+d
        xt = kk_proj(xt, obounds)
        try:
            ft, ifl, ict, tol = f(xt, h, core_data)
        except Exception as e:
            logger.warning('dropping function evaluation "%s"' % (str(e),))
            continue

        fct += ict
        single_point_hist_update(diff_hist, xt, ft, ifl)
        if least_squares == 1:
            frest = ft
            ft = numpy.norm(frest)/2.
        if ifl == 1:
            ft = fold+abs(fold)*.0001+1.E-12
        if ft < fval and aflag == 1:
            aflag, fval, fres, x = 0, ft, frest, xt
        if aflag == 1:
            lbda = beta*lbda
        iarm += 1

    if iarm == maxitarm and aflag == 1 and options.verbose == 1:
        logger.warning('line search failure %d %s', iarm, str(h))

    nfail = aflag
    return (fct, x, fval, iarm, aflag, fres, diff_hist, nfail)
