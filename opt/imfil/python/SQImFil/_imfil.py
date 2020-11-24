# Python version of IMFIL v1.02 "imfil.m" MATLAB version by C.T. Kelly.
#
# Modified and redistributed with permission.

from SQCommon import Result, ObjectiveFunction
from ._optset import optset
from ._util import error_check, check_first_eval, isok
from ._linalg import kk_proj, f_to_vals
from ._history import CompleteHistory, append_history, single_point_hist_update, \
   many_point_hist_update, scan_history
from ._exec_functions import gauss_newton, qn_update
import collections
import copy
import logging
import math
import numpy
import operator
import random

__all__ = ['minimize', 'log']

log = logging.getLogger('SKQ.ImFil')


#-----
class IterData(object):
    def __init__(self, h, obounds, itc, xb, fobjb, funsb, complete_history,
                 f_internal, core_data, options):
        self.h                = h
        self.obounds          = obounds
        self.itc              = itc
        self.xb               = xb
        self.fobjb            = fobjb
        self.funsb            = funsb
        self.complete_history = complete_history
        self.f_internal       = f_internal
        self.core_data        = core_data
        self.options          = options

    def copy(self):
        return IterData(self.h, self.obounds.copy(), self.itc, self.xb.copy(),
            self.fobjb.copy(), self.funsb, self.complete_history,
            self.f_internal, self.core_data, self.options)

class StencilData(object):
    def __init__(self, stencil_delta, svarmin, noise_val, bounds, v):
        self.stencil_delta = stencil_delta
        self.svarmin       = svarmin
        self.noise_val     = noise_val
        self.bounds        = bounds
        self.v             = v


#-----
def minimize(f, x0, bounds, budget=10000, optin=None, **optkwds):
    """
  Minimization of noisy functions subject to explicit bound
  constraints + hidden constraints

  opt_x, opt_f, histout, [complete_history] = imfil(x0, f, budget, bounds, optin)

  Input:
        x0 = initial iterate.
         f = objective function.
             The calling sequence for f should be
             fout, ifail, icount = f(x,h)

              h is an optional argument, and should be used only if your
              function is scale-aware, ie does something useful with
              the scale, such as tuning accuracy.

              fout = f(x)

              fail = 0 unless the evaluation of f fails, in which case
                     set ifail=1 and fout = NaN

              icount = number of calls to the expensive part of f
                      f should be smart enough to not do anything
                      if a bound constraint or a cheap hidden constraint
                      is violated

              f may return a fourth argument noise_level if f is
              noise aware. RTFM about this one.

    budget = max cost, uses icount from f(x)
             The iteration will terminate after the iteration that
             exhausts the budget. This argument is required.
             Suggestion: try 10*N^2.

  bounds   = [low, high] = N x 2 array of bound constraints on the variables.
             This is REQUIRED. If you want to solve an unconstrained
             problem, you must do so without the scaling we do in
             f_internal, and using imfil_core is currently the only way
             to do that. The lower bounds are the first column;
             the upper bounds are the second.

             I plan let you solve unconstrained problems by setting
             the bounds to Inf or -Inf. That will take some time while
             I figure out how I want to do the scaling (or make you
             do it).

   optin   = options structure
             Documentation on the way. In the meantime, RTFM or
             look at _optset.py.

 extra_arg = optional extra argument to be passed to f

  Output:
         x = estimated minimizer

   histout = iteration history, updated after each nonlinear iteration
           = (number of iterations )x(N+5) array, the columns are
              [fcount, fval, norm(sgrad), norm(step), iarm, xval]
              fcount      = cumulative function evals
              fval        = current function value
              norm(sgrad) = current projected stencil grad norm
              norm(step)  = norm of last step
              iarm        = line searches within current iteration
                          = -1 means first iterate at a new scale
              xval        = transpose of the current iteration

  complete_history = complete evaluation history

  complete_history is a structure with the data on every evaluation of f.

  complete_history.good_points has the successful points for columns.
  complete_history.good_values is a $M x N$ matrix  of the values at
               the good points. M > 1 for least squares.
               good_values = f(good_points)

  complete_history.failed_points has the unsuccessful points for columns.

  You may want to use this to build surrogates, decide to add new points to
  the stencil, or for troubleshooting. The complete history can take up a
  lot of room. I will only return it as output if you ask for it, and will
  not store it at all if you set the complete_history option to '0'.
  If I don't store it at all, then your functions to add directions to the
  stencil can't use that data.

  C. T. Kelley, July 19, 2010.
  This code comes with no guarantee or warranty of any kind."""

    global imfil_fscale

    log.debug('optimization start')

    # Make sure x0 is a single-column array
    if not isinstance(x0, numpy.ndarray):
        x0 = numpy.array(x0, shape=(len(x0), 1))
    elif len(x0.shape) == 1:
        x0 = x0.reshape(len(x0), 1)
    assert x0.shape[1] == 1

    # Make sure bounds are a 2-column array
    if not isinstance(bounds, numpy.ndarray):
        bounds = numpy.array(bounds)
        assert bounds.shape[1] == 2

    qbounds = bounds
    dbounds = bounds[:,(1,)] - bounds[:,(0,)]

    # Objective function should be callable
    if not callable(f):
        raise ValueError('"%s" is not callable' % (str(objfunc),))

    # Use optset to fill in defaults where no option values are provided
    if type(optin) == dict:
        options = optset(**dict(optin, **optkwds))
    else:
        options = optset(optin, **optkwds)

    objfunc = ObjectiveFunction(f, options)

    n = len(x0)

    # If the smooth_problem option is on, fix the dependencies and
    # change the scales.
    if options.smooth_problem == 1:
        log.info('smooth problem selected: adjusting options');
        bscales=[.5, .01, .001, .0001, .00001]
        options = optset(options,
              custom_scales=bscales,
              stencil_wins='yes',
              limit_quasi_newton='off',
              armijo_reduction=0.25,
              maxitarm=5)

    # Propagate 'verbose' option
    if options.verbose:
        log.setLevel(logging.INFO)

    # Load the fun_data structure so f_internal will know about the scaling.
    imfil_fscale = options.fscale
    FunData = collections.namedtuple('FunData', 'objfunc qbounds dbounds fun_fscale')
    fun_data = FunData(objfunc=objfunc,
        qbounds=qbounds, dbounds=dbounds, fun_fscale=imfil_fscale)
    CoreData = collections.namedtuple('CoreData', 'options fun_data')
    core_data = CoreData(options=options, fun_data=fun_data);

    # Scale the initial iterate to [0,1] before shipping to imfil_core.
    # imfil_core uses 0 and 1 as the bounds for optimization.
    error_check('bounds', x0, bounds)

    z0 = (x0 - qbounds[:,(0,)])/dbounds
    z, fval, histout, complete_history = \
       imfil_core(z0, f_internal, budget, core_data, bounds)

    # Unscale the results to original bounds.
    opt_par = numpy.multiply(dbounds, z) + qbounds[:,(0,)]
    db = numpy.diagflat(dbounds)
    qb = numpy.diagflat(qbounds[:,(0,)])
    histout = numpy.array(histout)
    xout = histout[:,5:n+5]
    xlow = numpy.ones(xout.shape).dot(qb)
    histout[:,5:n+5] = xout.dot(db) + xlow
    xout = histout[:,5:n+5]

    # Correct scaling for nonlinear least squares problems.
    if not options.least_squares:
        val_scale = imfil_fscale
    else:
        val_scale = math.sqrt(imfil_fscale)

    # Unscale the complete history
    if options.standalone and options.complete_history:
        ng, mg = len(complete_history.good_points), 2
        good_points = numpy.zeros((mg, ng))
        for i in range(ng):
            good_points[0, i] = complete_history.good_points[i][0]
            good_points[1, i] = complete_history.good_points[i][1]

        complete_history.good_values = [v*val_scale for v in complete_history.good_values]
        if mg > 0:
            clow = qb*numpy.ones((mg, ng))
            complete_history.good_points = db * good_points + clow

        nf = len(complete_history.failed_points)
        if nf > 0:
            failed_points = numpy.zeros((mf, nf))
            for i in range(nf):
                failed_points[:,(i)] = complete_history.failed_points[i]
            flow = qb * numpy.ones((mf, nf))
            complete_history.failed_points = db * failed_points + flow

    # Unscale the function and gradients.
    histout[:,1] = histout[:,1]*imfil_fscale
    histout[:,2] = histout[:,2]*imfil_fscale

    # Turn opt_par into an array (are a matrix) and scale minimum.
    result = Result(fval*val_scale, numpy.squeeze(numpy.asarray(opt_par)))

    if options.standalone:
        if options.complete_history:
            return result, histout, complete_history
        return result, histout
    return result, objfunc.get_history()


#-----
def imfil_core(x0, f, budget, core_data, bounds):
    """
  Minimization of f(x) subject to explicit bound constraints

  x, fval, histout, complete_history = \
          imfil_core(x0, f, budget, core_data, bounds)

  Bound constrained, parallel, implicit filtering code.

  IMPLICIT FILTERING with SR1 and BFGS quasi-Newton methods

  Input:
        x0 = initial iterate
         f = objective function,

             the calling sequence for f should be
             [fout,ifail,icount]=f(x,h)

             h is an optional argument for f, and should be used only if your
             function is scale-aware, ie does something useful with
             the scale, such as tuning accuracy.

   budget = upper limit on function evaluations. The optimization
                  will terminate soon after the function evaluation counter
                  exceeds the budget.

   core_data = structure with the options + fun_data
               The options are documented in optset.m.
               fun_data is private to imfil.m and is used in the internal
               scaling for the function evaluation.

   bounds  = N x 2 array of bound constraints on the variables.
             These are the original bounds for the problem. We only
             use these if you use the add_new_directions option.

  Output:
         x = estimated minimizer
      fval = corresponding minimum
   histout = iteration history, updated after each nonlinear iteration
           = N+five column array, the rows are
             [fcount, fval, norm(sgrad), norm(step), iarm, xval]
             fcount = cumulative function evals
             fval = current function value
             norm(sgrad) = current (projected) simplex grad norm
                         = -1 means no gradient for this xval, this
                  can happen, for example, if you hit the target before you
                  evaluate the simplex derivative
             norm(step) = norm of last step
                        = 0 means no change, eg for stencil failure
             iarm=line searches in current iteration to move to new point
                 =-1 means first iterate at a new scale or that the
                     inner iteration was terminated before the line search,
                     eg for stencil failure
              xval = transpose of the current iteration

  complete_history = complete evaluation history

  complete_history is a structure with the data on every evaluation of f.
  complete_history.good_points has the successful points for columns.
  complete_history.good_values is a $M x N$ matrix  of the values at
               the good points. M > 1 for least squares.
               good_values = f(good_points)

  complete_history.failed_points has the unsuccessful points for columns.

  You may want to use this to build surrogates, decide to add new points to
  the stencil, or for troubleshooting. The complete history can take up a
  lot of room. I will only return it as output if you ask for it, and will
  not store it at all if you set the complete_history option to '0'.
  If I don't store it at all, then your functions to add directions to the
  stencil can't use that data.

  Raises exception in case of failure.

  C. T. Kelley, July 16, 2010
  This code comes with no guarantee or warranty of any kind."""

    global imfil_fscale

    fcount = 0
    options = core_data.options
    # set up the difference scales and options
    n = len(x0)

    # In imfil_core the bounds are 0 and 1 on all the variables.
    # We use the real bounds only if you use the add_new_directions
    # option.
    obounds = numpy.zeros((n, 2)); obounds[:,(1,)] = 1

    # Get the options we need.
    imfil_maxit    = options.maxit
    imfil_target   = options.target
    imfil_termtol  = options.termtol
    stencil_wins   = options.stencil_wins
    complete_history = CompleteHistory()
    assert not complete_history.good_values

    # The explore option may require some setup.
    explore_function, explore_data_flag, explore_data = setup_explore(options)

    # Initialize the iteration; create the stencil; set up the scales; ...
    x = x0.copy(); xold = x0.copy(); histout = []
    dscal = create_scales(options); nscal = len(dscal)
    if options.executive == 1:
        hess = options.executive_data
    else:
        hess = numpy.eye(n)

    xc = x0.copy(); ns = 0; failc = 0
    stop_now = False
    fval = imfil_target+1

    # Sweep through the scales.
    sflag = 1
    while ns < nscal and fcount <= budget and failc < options.maxfail and \
           not stop_now and fval > imfil_target:
        h = dscal[ns]
        log.debug('current scale: %f', h)

        # Evaluate the function to test for instant termination and (if
        # noise_aware is on) to get the estimate of the noise. Both the noise
        # and the value of f  may vary as a function of the scale, so we
        # have to test it every time.
        #
        # We could move this outside of the main loop if noise_aware and scale_aware
        # are both off. In any case, we don't have to reevaluate f after a stencil
        # failure unless f is scale-aware.
        if options.noise_aware > 0 or fcount == 0 or options.scale_aware > 0:
            funs, iff, icf, noise_val = f(x, h, core_data)
            fval = f_to_vals(funs, options.least_squares)
            icount = icf
        else:
            icount = 0

        # The first call to f is the place to check for sanity.
        #
        # The first call to f also sorts out the scaling. So you can't scale the
        # the targets, errors, or deltas before that call.
        if fcount == 0:
            error_check('first_eval', funs, options, iff)

            imfil_target = imfil_target/imfil_fscale
            function_delta = options.function_delta/imfil_fscale
            append_history(histout, 1, fval, 0, 0, 0, x)

            # Initialize the internal data structures.
            itc = 0
            stencil_data = \
                create_stencil_data(options, imfil_fscale, noise_val, bounds)
            iteration_data = IterData(h=h, obounds=obounds, itc=itc,\
               xb=x, fobjb=fval, funsb=funs, complete_history=complete_history,\
               f_internal=f, core_data=core_data, options=options)
        else:
            # End of first-call-to-f block
            #
            # If it's not the first call to f, update the internal data structures.
            iteration_data.h = h
            stencil_data.noise_val = noise_val;

        if options.complete_history:
            single_point_hist_update(iteration_data.complete_history, x, funs, iff)

        fcount += icf
        if fval < imfil_target :
            stop_now = True
            break

        stol = imfil_termtol*h; iarm=0; nfail=0

        # Compute the stencil gradient to prepare for the quasi-Newton iteration.
        sdiff, sgrad, npgrad, fcount, sflag, jac, iteration_data, stop_now = \
            manage_stencil_diff(x, f, funs, \
                  iteration_data, fcount, stencil_data)
        append_history(histout, fcount, fval, npgrad, -1, -1, x)
        gc = sgrad

        #
        if npgrad < stol or sflag == 0 or stop_now:
            # Declare convergence at this scale on stencil failure or tolerance match.
            if npgrad < stol:
                log.debug('converged at current scale with gradient (%f) below cut-off (%f)',
                          npgrad, stol)
            else:
                log.debug('ended current scale due to stencil failure')
            gc = sgrad
            if sflag != 0:
                failc += 1
            else:
                failc = 0
        else:
            # Take a few quasi-Newton iterates. This is the inner iteration.
            log.debug('start inner Newton loop iteration')
            failc = 0

            # itc = inner iteration counter
            itc = 0

            # Newton while loop
            while itc < imfil_maxit*n and fval > imfil_target and \
                  npgrad >= stol and nfail==0 and fcount < budget and sflag > 0:

                itc += 1
                iteration_data.itc = itc;
                fc = fval
                # Take an inner iteration. sdiff is the simplex derivative
                # (gradient or Jacobian) at x. gc is the simplex gradient at xc.
                xp, fval, funs, fcount, iarm, iteration_data, nfail, hess = \
                    inner_iteration(f, x, funs, sdiff, xc, gc, \
                                    iteration_data, hess, fcount)

                # Stop the entire iteration if you've hit the target.
                if fval < imfil_target:
                    stop_now = True
                    x = xp
                    stepn = numpy.linalg.norm(xold-x, ord=numpy.inf)
                    append_history(histout, fcount, fval, npgrad, stepn, iarm, x)
                    break

                # Update xold, xc, and x. At this point the model Hessian and gc
                # are evaluated at xc. xc and gc only get updated right here.
                xold = x.copy(); xc = x.copy(); gc = sgrad; x = xp.copy()

                # If stencil_wins is on, then take the best point you have.
                # You'll update x, but not xc.
                if stencil_wins == 1:
                    x, funs, fval = write_best_to_x(iteration_data)
                    stepn = numpy.linalg.norm(xc-x, ord=numpy.inf)
                    append_history(histout, fcount, fval, npgrad, stepn, iarm, x)
                    nfail = 0

                # Stop on small objective function changes?
                fdelta = abs(fval-fc)
                if fdelta > 0 and fdelta < function_delta :
                    stop_now = True
                    sflag = 0
                    stepn = numpy.linalg.norm(xc-x, ord=numpy.inf)
                    append_history(histout, fcount, fval, npgrad, stepn, iarm, x)
                    break

                #    Compute the difference gradient for the next nonlinear iteration.
                if not stop_now and fcount < budget:
                    sdiff, sgrad, npgrad, fcount, sflag, jac, iteration_data, stop_now = \
                        manage_stencil_diff(x, f, funs,
                            iteration_data, fcount, stencil_data)

                # If the quasi-Newton method terminated successfully or with a
                # stencil failure, make sure x is now the best point.
                if stop_now or sflag == 0:
                    iteration_data, rflag = \
                        reconcile_best_point(funs, xp, iteration_data);
                    x, funs, fval = write_best_to_x(iteration_data)
                    stepn = numpy.linalg.norm(xc-x, ord=numpy.inf)
                    append_history(histout, fcount, fval, npgrad, stepn, -1, x)
                    break

                # Otherwise, update the history array and keep going.
                else:
                    stepn = numpy.linalg.norm(xc-x, ord=numpy.inf)
                    append_history(histout, fcount, fval, npgrad, stepn, iarm, x)

                # Update xold. It will not be the same as xc (the point for the most
                # recent stencil derivative point) if you are exiting the quasi-Newton
                # loop.
                xold = x.copy()

                # end Newton while loop

            log.debug('end inner Newton loop iteration')
        # end test for stencil failure for first derivative at new scale

        # Apply the explore_function if there is one.
        if options.explore == 1:
            fcount, iteration_data = \
                manage_explore(explore_function, f, iteration_data, fcount, explore_data)

        # After the quasi-Newton iterate, I make sure that the
        # quasi-Newton point is the best I've seen. If it's not, I fix it.
        # So, 'stencil_wins' is half-way on. One consequence of this is that
        # I take the best point if the inner iteration fails.
        iteration_data, rflag = reconcile_best_point(funs, x, iteration_data)
        x, funs, fval = write_best_to_x(iteration_data)

        if rflag == 1:
            if histout[-1][0] == fcount:
                histout = histout[:-1]

        append_history(histout, fcount, fval, npgrad, 0, 0, x)

        if options.verbose:
            log.info(ns, fval, h, npgrad, itc, fcount)

        ns += 1
        # end of loop over scales

    return x, fval[0], histout, complete_history


#-----
def f_internal(x, h, core_data):
    """
  Creates an O(1) dummy function with unit bounds.
  So, (I hope) the gradient and stencil are reasonably scaled.

  I encode scale and noise awareness in this function, too. If your
  function is not noise aware, I pass a zero noise back to imfil_core.
  imfil_core will do the right thing after that."""

    global imfil_fscale

    options = core_data.options
    exarg = options.extra_arg_value
    fun_data = core_data.fun_data
    objfunc = fun_data.objfunc
    qbounds = fun_data.qbounds
    dbounds = fun_data.dbounds

    mx, nx = x.shape
    z = x.copy().reshape(mx, nx)
    for ix in range(nx):
        z[:,(ix,)] = numpy.multiply(dbounds, x[:,(ix,)])+qbounds[:,(0,)]
    zargs = numpy.array(numpy.squeeze(z.T))

    call_args = [zargs,]
    if options.scale_aware:
        call_args.append(h)
    if options.extra_argument:
        call_args.append(exarg)

    res = objfunc(*call_args)
    fx, iff, icf, tol = 0., 0, 1, 0.
    if type(res) == tuple:
        fx = res[0]
        if 1 < len(res):
            iff = res[1]
        if 2 < len(res):
            icf = res[2]
        if 3 < len(res):
            tol = res[3]
    else:
        fx = res
        mz, nz = z.shape
        if nz != 1:
           iff = numpy.zeros(nz); icf = nz*numpy.ones(nz)

    # If this is the first time you evalute f and if imfil_fscale < 0,
    # we change imfil_fscale to the correct relative scaling factor. This
    # also makes imfil_fscale > 0 so we only compute the scaling once.
    if imfil_fscale <= 0:
        if imfil_fscale == 0:
            imfil_fscale = -1.2

        if options.least_squares:
            val = fx[:,0].dot(fx[:,0])/2.
            scale_base = val
        else:
            scale_base = abs(fx)

        if scale_base == 0:
            imfil_fscale = 1
        else:
            imfil_fscale = abs(imfil_fscale)*scale_base

    # Scale the function value.
    if not options.least_squares:
        fx /= imfil_fscale;
    else:
        fx /= math.sqrt(imfil_fscale)
    tol /= imfil_fscale

    return fx, iff, icf, tol


#-----
def setup_explore(options):
    """
  Gets the explore function organized if you have one."""

    explore_function = []
    explore_data_flag = 0
    explore_data = []
    if options.explore:
        explore_function  = options.explore_function
        explore_data_flag = options.explore_data_flag
    if explore_data_flag:
        explore_data = options.explore_data

    return explore_function, explore_data_flag, explore_data


#-----
def create_scales(options):
    """
  Creates the scales for imfil.m. Nothing much here right now, but
  custom scales ... are in the near future.

  dscal = create_scales(options)

  C. T. Kelley, September 15, 2008
  This code comes with no guarantee or warranty of any kind."""

    custom_scales = options.custom_scales
    mcs = len(custom_scales)
    if mcs > 0:
        dscal = custom_scales
    else:
        step  = options.scale_step
        start = options.scale_start
        depth = options.scale_depth

        # Do some error checking. Warn or complain as needed.
        if start > depth:
            raise ValueError('imfil_create_scales: error in scales, start > depth')

        # scale reduces with power of 2
        dscal = [step**-x for x in range(start, depth+1)]

    return dscal


#-----
def create_stencil_data(options, fscale, noise_val, bounds):
    """
  Build a structure with the data, targets, errors, and deltas that
  you need to compute the stencil derivative and  determine stencil failure.
  This structure gets passed to manage_stencil_diff and contains everything
  that depends only on the scale and the options. I'm doing this mostly
  to keep the argument list from occupying several lines and confusing me.

  stencil_data = create_stencil_sdata(options, imfil_fscale, noise_val, bounds)

  Do not attempt to land here."""

    n = len(bounds[:,1])
    v = create_stencil(options, n)
    stencil_delta = options.stencil_delta/fscale
    svarmin       = options.svarmin/fscale

    stencil_data = StencilData(
        stencil_delta=stencil_delta, svarmin=svarmin,
        noise_val=noise_val, bounds=bounds, v=v)
    return stencil_data


#-----
def create_stencil(options, n):
    """
  Builds the stencil for _imfil.py. As we evolve this we will be
  adding the ability to do random rotations for all or part of
  a stencil and all sorts of other stuff.

  vstencil = create_stencil(options, n)

  Input:
         options = imfil.m options structure
               n = dimension

  Output:
               v = stencil

  C. T. Kelley, July 14, 2009
  This code comes with no guarantee or warranty of any kind."""

    stencil  = options.stencil
    vstencil = options.vstencil

    # Is vstencil really there?
    if vstencil:
        mv, nv = vstencil.shape
        if mv+nv > 0:
            stencil = -1

    # Build the stencil.
    if stencil == -1:    # Custom stencil is ok.
        pass
    elif stencil == 0:   # central difference stencil
        v = numpy.hstack((numpy.eye(n), -numpy.eye(n)))
    elif stencil == 1:   # one sided stencil = ON THE WAY OUT!
        v = numpy.eye(n)
    elif stencil == 2:   # positive basis stencil
        v = numpy.hstack((numpy.eye(n), -numpy.ones(n)/math.sqrt(n)))
    else:
        raise ValueError('illegal stencil in _imfil.create_stencil')

    # If vstencil is a custom job, take it. Otherwise use one of
    # the internal choices.
    if stencil != -1:
        vstencil = v

    return vstencil


#-----
def stencil_diff(x, f, dx, fc, iteration_data, complete_history):
    """
  Stencil derivative for optimization and nonlinear least squares.
  This function also tests for best point in stencil.

  grad, best_value, best_value_f, best_point, icount, sflag,svar, diff_hist, jac = \
      stencil_diff(x, f, dx, fc, iteration_data, options, h, complete_history)

    Input:  x  = current point
            f  = objective function (or vector residual for
                 nonlinear least squares)
            dx = scaled difference directions
            fc = current function value

    iteration_data = structure with the options + fun_data +
                     current iteration parameters.
                The options are documented in optset.m.
                fun_data is private to imfil.m and is used in the internal
                scaling for the function evaluation.

  complete_history = every point imfil has ever seen. That data are used
                     here to prevent redundant evaluations.

    Output:
          grad = stencil gradient; NaN if every point on the stencil is bad
   best_value  = best value in stencil
   best_value_f= best least squares residual in stencil
   best_point  = best point in stencil
        icount = counter of calls to expensive part of function
                  the function needs to return this.
         sflag = 0 current point is best in the stencil
                      This means stencil failure.

               = 1 means there's a better point on the stencil.

          svar = variation on stencil = max - min

     diff_hist = every function evaluation for points satisfying the
                  bounds stored in a nice struct. This is used to update
                  the complete_history structure.

           jac = stencil Jacobian for least squares problems.

  C. T. Kelley, July 30, 2009
  This code comes with no guarantee or warranty of any kind."""

    core_data = iteration_data.core_data
    h = iteration_data.h

    # bounds = iteration_data.obounds are the 0-1 bounds for the
    # scaled problem.
    bounds = iteration_data.obounds

    options = core_data.options
    least_squares = options.least_squares

    # Poll the stencil and collect some data.
    best_value, best_value_f, best_point, icount, sgood, good_points, good_values,\
         good_dx, good_df, failed_points = \
           poll_stencil(x, f, dx, fc, bounds, core_data, h, complete_history)

    # Initialize the search for best value at the center.
    if least_squares:
        fval = fc.dot(fc)/2   # best value in the stencil
    else:
        fval = fc

    # Use the points and values to get the stencil statisitcs.
    sflag, best_value, best_value_f, best_point, svar, diff_hist = \
        collect_stencil_data(good_points, good_values, failed_points,
           x, fval, fc, fc, options)

    # sgood = 0 means there are no good points. Shrink time!
    #
    # Estimate the derivative.
    # The idea is that fprime*dx approx df, where
    # fprime   = m x n is the gradient-transpose or the Jacobian
    # good_dx  = n x pnew (pnew <= vsize) is the matrix of good steps
    # good_df  = m x pnew is the collection of good function values at x + dx
    #
    # So the least squares problem to be solved is
    #         min || fprime * good_dx - good_df ||
    # for the columns of fprime. The minimum norm solution is
    # fprime = good_df*pinv(good_dx).
    if sgood > 0:
        pdx = numpy.linalg.pinv(good_dx)
        fprime = good_df.dot(pdx)
        if least_squares == 0:
            grad = fprime.T
            jac=[]
        else:
            jac = fprime
            grad = fprime.dot(fc)
    else:
        grad = numpy.NaN*x
        jac = []

    return grad, best_value, best_value_f, best_point, icount, sflag, svar, diff_hist, jac


#-----
def manage_stencil_diff(x, f, funs, iteration_data, fcount, stencil_data):
    """
  This function calls stencil_diff to compute the stencil derivative,
  updates the evaluation counter and complete_history, runs through
  all the tests for stencil failure, and sends back everything
  imfil_core needs to do its job.

  This is an internal function. There is no reason to hack this code.
  Don't do it.

  This function is under development and changes frequently.

  C. T. Kelley, January 12, 2011"""

    options = iteration_data.options

    h = iteration_data.h

    # Unpack the stencil_data structure.
    stencil_delta = stencil_data.stencil_delta
    svarmin       = stencil_data.svarmin
    noise_val     = stencil_data.noise_val
    obounds       = iteration_data.obounds
    bounds        = stencil_data.bounds
    v             = stencil_data.v

    # Complete the direction matrix and compute the stencil derivative.
    vv = augment_directions(x, v, h, options, bounds)
    complete_history = iteration_data.complete_history
    sgrad, fb, fbf, xb, icount, sflag, svar, diff_hist, jac = \
        stencil_diff(x, f, h*vv, funs, iteration_data, complete_history)
    fcount += icount
    pgrad = x - kk_proj(x-sgrad, obounds)
    npgrad = numpy.linalg.norm(pgrad, ord=numpy.inf)

    # Run the optional stencil failure tests.
    #
    # If noise_aware = 1 (i.e. noise_val returned) and the scaled variation
    # in f is < than the function's estimate of the noise, then I declare
    # stencil failure!
    if max(noise_val, svarmin) > svar:
        log.debug('scaled variation (%f) is less than estimated noise (%f)', svar, noise_val)
        sflag = 0

    # If stencil_delta > 0, then I terminate the entire optimization
    # when the scaled variation in f < stencil_delta. I report this as
    # stencil failure as well.
    stop_now = False
    if stencil_delta > svar:
        log.debug('scaled variation (%f) is less than stencil delta (%f)', svar, stencil_delta)
        stop_now = True
        sflag = 0

    # Update the iteration_data structure.
    iteration_datap = iteration_data
    if options.complete_history:
        complete_history = many_point_hist_update(complete_history, diff_hist, False)
        iteration_datap.complete_history = complete_history

    iteration_datap, rflag = reconcile_best_point(fbf, xb, iteration_datap);

    sdiff = jac_or_grad(sgrad, jac, options)

    return sdiff, sgrad, npgrad, fcount, sflag, jac, iteration_datap, stop_now


#-----
def poll_stencil(x, f, dx, fc, bounds, core_data, h, complete_history):
    """
  Poll the stencil and organize the results.
  I do not think you will want to modify this code.

  best_value, best_point, icount, sgood, good_points, good_values, \
          good_dx, good_df, failed_points = \
      poll_stencil(x, f, dx, fc, bounds, core_data, h, complete_history)

  Input:
           x = center of stencil
           f = objective function
          dx = h*V = array of scaled directions
          fc = f(x)
      bounds = N x 2 array of lower/upper bounds
   core_data = imfil core_data structure
           h = current scale

  Output:

  The best_point data is frozen at xc for this function. We only keep
  track of it here so we can ship it to
      best_point    = f(best_point) == best_value
      best_value    = lowest value of f on the stencil
      best_value_f  = best least squares residual
      icount        = cost in units of calls to f
      sgood         = number of successful evaluations
      good_points   = list of points at which f returned a value
      good_values   = f(good_points)
      good dx       = vector of good_points - x
      good df       = vector of f(good_points) - fc
      failed_points = list of points at which f did not return a value

  C. T. Kelley, Aug 13, 2009
  This code comes with no guarantee or warranty of any kind."""

    options = core_data.options
    least_squares = options.least_squares

    n, vsize = dx.shape

    # Record the best point and best function value in the stencil.
    # Get started by using the center point.
    best_point = x
    if least_squares:
        best_value = fc.dot(fc)/2  # best value in the stencil
    else:
        best_value = fc
    best_value_f = fc

    iflag = numpy.zeros((vsize, 1))

    try:
        m = len(fc)
    except TypeError:
        m = 1

    good_points = []
    good_values = []
    fval = numpy.zeros(vsize)
    icount = 0

    # fp[:,i] and fval(i) are not defined outside of the bounds
    # or if iflag(i) == 1.
    #
    # First cull the points which violate the bound constraints.
    #
    # Collect the feasible points, differences, and functions in xp1 and dx1.
    dx1 = None
    xp1 = None
    xp  = numpy.zeros(dx.shape)

    # One-sided differences may need to flip the direction if the positive
    # perturbation is not feasible.
    if options.stencil == 1:
        for i in range(vsize):
            xp[:,i] = x+dx[:,(i,)]
            if isok(xp[:,i], bounds) == 0:
                dx[:,(i,)] = -dx[:,(i,)]

    for i in range(vsize):
        xp[:,(i,)] = x+dx[:,(i,)]
        if isok(xp[:,i], bounds):
            if dx1 is None:
                dx1 = dx[:,(i,)]
                xp1 = xp[:,(i,)]
            else:
                dx1 = numpy.hstack((dx1, dx[:,(i,)]))
                xp1 = numpy.hstack((xp1, xp[:,(i,)]))

    npoints = xp1.shape[1]
    fp = numpy.zeros((m, npoints))

    # Query the complete_history structure to see if you've evaluated f
    # at any of these points before.
    oldresults, newpoints = scan_history(complete_history, xp1, fp, dx1)

    # Copy over pre-existing values.
    iflago = numpy.zeros((npoints, 1))
    for key, val in oldresults.items():
        fp[:,(key,)] = val[0]
        iflago[i]    = val[1]

    # Evaluate f, in parallel if possible. Flag the failed points.
    if not options.parallel:
        fp1, iflag = [], []
        for point in newpoints.values():
            fpx, iflagx, ict, tol = f(point, h, core_data)
            fp1.append(fpx)
            iflag.append(iflagx)
            icount += ict
    elif newpoints:
        fp1, iflag, ictrp, tol = f(numpy.hstack(tuple(newpoints.values())), h, core_data)
        icount += sum(ictrp)

    fp = numpy.zeros((1, len(fp1)))

    # Copy over new values.
    if newpoints:
        for i, key in enumerate(newpoints.keys()):
            fp[:,(key,)] = fp1[i]
            iflago[key]  = iflag[i]

    fp1 = fp.T; iflag = iflago

    # Identify the failed points.
    ibad = numpy.nonzero(iflag == 1)[0]
    failed_points = numpy.zeros((len(ibad), n))
    if len(ibad):
        for idx, key in enumerate(ibad):
            failed_points[(idx,),:] = newpoints[key].T

    # Store the good points and their function values.
    igood = numpy.nonzero(iflag == 0)[0]
    sgood = len(igood)
    good_df = numpy.zeros(fp1.shape)
    good_dx = []
    if sgood > 0:
        good_points = xp1[:,igood]
        if least_squares == 1:
            good_fp = fp1[:,igood]
        else:
            good_fp = fp1[igood]
        good_dx = dx1[:,igood]
        for ig in range(sgood):
            if least_squares == 1:
                good_df[:,ig] = good_fp[:,ig] - fc
            else:
                good_df[ig] = good_fp[ig] - fc
        good_values = good_fp

    return (best_value, best_value_f, best_point, icount, sgood, \
        good_points, good_values, good_dx, good_df.T, failed_points)


#-----
def collect_stencil_data(good_points, good_values, failed_points,
        best_point_old, best_value_old, best_value_f_old, fc, options):

    least_squares = options.least_squares

    # Who's number one?
    good_scalars = f_to_vals(good_values, least_squares)
    ibest, xbest_value = min(enumerate(good_scalars), key=operator.itemgetter(1))
    xbest_point = good_points[:,ibest]
    if least_squares:
        xbest_value_f = good_values[:,ibest]
    else:
        xbest_value_f = good_values[ibest]
    if xbest_value < best_value_old:
        best_value = xbest_value
        best_value_f = xbest_value_f
        best_point = xbest_point
    else:
        best_value = best_value_old
        best_value_f = best_value_f_old
        best_point = best_point_old

    # Find the big loser.
    worst_value = max(good_scalars)

    # Assemble the history structure.
    DiffHist = collections.namedtuple('DiffHist', 'good_points good_values failed_points')
    diff_hist = DiffHist(good_points=good_points,
                    good_values=good_values, failed_points=failed_points)

    # What's the total variation?
    svar = worst_value-best_value

    # Stencil failure?
    sflag = 1
    if abs(best_value_old-best_value) < 1.E-14:
        global imfil_fscale
        log.debug('no improvement found on stencil (current best: %f)',
                  imfil_fscale*best_value_old)
        sflag = 0
        jac = []
        grad = []

    if len(best_point.shape) == 1:
        best_point = best_point.reshape((len(best_point), 1))
    return sflag, best_value, best_value_f, best_point, svar, diff_hist


#-----
def reconcile_best_point(funs, x, old_data):
    """
  new_data, rflag = reconcile_best_point(funs, x, old_data)

  After the poll, or when it's time to terminate the inner or outer
  iteration, you may have a new best point.
  This function updates the record of the best point using the
  iteration_data structure.

  Input: x        = current point
         funs     = f(x)
     old_data     = iteration_data structure
   least_squares  = Are we solving a nonlinear least squares problem?

  Output: new_data     = updated iteration_data structure
          fvalout = f(xout)
          rflag = 0 if xout = x;  (ie f(x) is best, new best point)
          rflag = 1 if xout = xb; (best point unchanged)"""

    new_data = old_data.copy()

    rflag = 1
    fb = old_data.fobjb
    fval = f_to_vals(funs, old_data.options.least_squares)
    if fval < fb:
        rflag = 0
        new_data.xb    = x
        new_data.funsb = funs
        new_data.fobjb = fval

    return new_data, rflag


#-----
def inner_iteration(f, x, fx, sdiff, xc, gc, iteration_data, hess, fcount):
    """
  Manage the various inner iteration options.

  Inputs:
          f = objective function; f returns an residual vector.
          x = current point.
        fun = current (vector) residual at x.
        jac = stencil Jacobian at x.
         xc = previous point; xc is not used in this function, but
              that may change if we elect to put a quasi-Newton
              method in here. xc=x for the first iteration in the
              inner iteration.
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
       iarm = number of step size/trust region radius reductions.
  diff_hist = history data for the Gauss-Newton loop
      nfail = 0 if the line search/trust region/LM succeeds, 1 if it fails"""

    options          = iteration_data.options
    core_data        = iteration_data.core_data
    least_squares    = options.least_squares
    obounds          = iteration_data.obounds
    opt_exec         = options.executive
    if opt_exec:
        opt_exec_function = options.executive_function

    if not opt_exec:
        if not least_squares:
            xp, fval, funs, fct, iarm, diff_hist, nfail, hess = \
                qn_update(f, x, fx, sdiff, xc, gc, iteration_data, hess)
        else:
            xp, fval, funs, fct, iarm, diff_hist, nfail, hess = \
                gauss_newton(f, x, fx, sdiff, xc, gc, iteration_data, hess)
    else:
        xp, fval, funs, fct, iarm, diff_hist, nfail, hess = \
            opt_exec_function(f, x, fx, sdiff, xc, gc, iteration_data, hess)

    fcount += fct
    iteration_datap = iteration_data
    if options.complete_history:
        many_point_hist_update(iteration_data.complete_history, diff_hist)

    iteration_datap, rflag = reconcile_best_point(funs, xp, iteration_datap)
    return (xp, fval, funs, fcount, iarm, iteration_datap, nfail, hess)


#-----
def write_best_to_x(iteration_data):
    """
  Makes the best point the current iterate.

  x, funs, fval = write_best_to_x(iteration_data)"""

    x    = iteration_data.xb
    funs = iteration_data.funsb
    fval = iteration_data.fobjb
    return x, funs, fval


#-----
def augment_directions(x, vin, h, options, bounds):
    """
  Enrich the stencil with random directions and a user-supplied
  new_direcions function.

  C. T. Kelley, August 13, 2009.
  This code comes with no guarantee or warranty of any kind.

  Add the random directions."""

    vout = random_augment(vin, options.random_stencil)

    # See if there's a new_directions function.
    new_directions = options.add_new_directions
    if new_directions:
        dbv = bounds[:,(1,)] - bounds[:,(0,)]; db = numpy.diagflat(dbv)
        unscaled_x = db*x + bounds[:,(0,)]
        unscaled_v = db*vout
        unscaled_vnew = new_directions(unscaled_x, h, unscaled_v)
        mv, nv = unscaled_vnew.shape
        if mv > 0:
            vnew = numpy.linalg.inv(db)*unscaled_vnew;
        mv, nv = vnew.shape
        for i in range(nv):
            vnew[:,i] = vnew[:,i]/numpy.linalg.norm(vnew[:,i], ord=numpy.inf)
        vout = numpy.hstack((vout, vnew))

    return vout


#-----
def random_augment(vin, k):
    """
  Add k random unit vectors to the stencil.

  This makes some of the theory work, but it is not always
  a good idea to do this. Adding more vectors makes stencil failure
  less likely, which is not good if you have a full stencil and you're
  wasting lots of time in the line search.

  Adding random vectors makes more sense if the minimizer is
  on a hidden or explicit constraint boundary.

  C. T. Kelley, January 5, 2011
  This code comes with no guarantee or warranty of any kind."""

    n = len(vin)
    if k == 0:
        vout = vin  # TODO: why? seems a dead assignment

    # Generate k uniformly distributed points on the unit sphere.
    # See Marsaglia, G. "Choosing a Point from the Surface of a Sphere."
    #     Ann. Math. Stat. 43, 645-646, 1972.
    rv = numpy.random.normal(0., 1., (n, k))
    for ir in range(k):
        rv[:,ir] /= numpy.linalg.norm(rv[:,ir])

    if rv.any():
        vout = numpy.hstack((vin, rv))

    return vout


#-----
def jac_or_grad(sgrad, jac, options):
    """
  Returns either the simplex gradient or Jacobian depending
  on the type of problem (optization or least squares).

  sdiff = jac_or_grad(sgrad, jac, options)"""

    least_squares = options.least_squares
    if least_squares == 1:
        sdiff = jac
    elif least_squares == 0:
        sdiff = sgrad
    else:
        raise ValueError('error in jac_or_grad')
    return sdiff
