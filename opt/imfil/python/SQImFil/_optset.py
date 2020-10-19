import copy, string

__all__ = ['optset']

STANDALONE = True

def optset(optin=None, **optkwds):
    """
  Set the options in imfil.
  C. T. Kelley, Nov 11, 2007

  optout = optset(name1=val1, name2=val2, ...)
           creates the options dictionary with the named options
           set to something other than the defaults.

  optout = optset(optin, name1=val1, name2=val2, ...);
           modifies the dictionary <optin> and creates a new structure
           <optout>.

  And now for the lists. The details of the options, the motiviation
  for the defaults, and suggestions for their use live in the manual.

  Scalar Options:

  armijo_reduction: Reduction factor for the linesearch. The default is .5.

  complete_history: 1 or 'on' to store complete history of all calls to f. 0
         or 'off' otherwise. The default is 'off'. Do not turn this off unless
         you are having severe problems with storage.

  fscale: Sets a "typical value of f", used to improve the quality of the
         difference gradient. The default is imfil_fscale = 1.2*| f(x_0) |.
         Setting options = optset(options, fscale=-c); with c > 0 will set
         imfil_fscale = c * |f(x_0)| and options = optset(options, fscale=c)
         with c > 0 will set imfil_fscale = c.

  function_delta: Set this to an estimate for the absolute error in f if you
         have one. Ignore this option if you do have no idea how accurate your
         function evaluation is. Set function_delta > 0 and the optimization
         will terminate if two successful iterations (ie. iterations that
         change the best point) have function values that differ
         by < function_delta. The default is -1, which means that
         function_delta does not affect the iteration.

  stencil_delta: Set this to an estimate for the absolute error in f if you
         have one. Ignore this option if you do have no idea how accurate your
         function evaluation is. The optimization loop will terminate when the
         maximum absolute difference of function values on the stencil
         is < stencil_delta. The default is -1, which means that stencil_delta
         has no effect.

  least_squares: imfil does the smart thing if you're solving a nonlinear
         overdetermined (m >=n) least squares problem. Set least_squares=1 and
         let your function return the vector residual in R^m. imfil will
         compute f'*f/2 as part of the optimization.

  limit_quasi_newton: 'yes' or 1 means that the quasi-Newton direction will be
         truncated to a length of at most 10*h, where h is the current scale.
         This is a good idea if the function is very noisy, as it can keep the
         line search from thrashing about. The code is different from the old
         FORTRAN version, and now I am not sure about what the best value is.
         'yes' is currently the default. 'no' or 0 means the quasi-Newton
         direction vector gets left alone.

  maxit: The quasi-Newton inner iteration is limited to maxit*n iterations
         for each scale. The default is 50.

  maxitarm: The line search will reduce the step by at most maxitarm times
         before declaring failure. The default is 3. The line search is on a
         short leash for good reason. If you don't find something useful after
         three reductions, you're not likely to do better with more effort.

  maxfail: After maxfail line search or stencil failures, imfil stops
         the entire iteration. The default is maxfail = 3.

  noise_aware: Set this to one if your function can return as estimate of the
         norm of the noise. The call is [fout,ifail,icount,noise_val]=f(x).
         In this case imfil will declare stencil failure if the difference
         between the max and min values on the stencil is < noise_val.
         The default is 0 (not noise_aware).
         You can set scale_aware=1 and noise_aware=1 at the same time.

         noise_val is not the same as stencil_delta. You set noise_val via the
         call to f as f's best estimate of the error. noise_val is used as a
         test to reject the stencil and reduce the scale.
         stencil_delta is used to terminate the optimization.

  parallel: Set this to one if f can accept multiple input vectors and
         exploit parallelism. The output should be vectors of
         [fout,ifail,icount], with a length equal to the number of input
         variables. When imfil computes the stencil gradient f will get all
         the directions at once. It is then your job to manage the parallel
         computation for optimal load balancing.

  quasi: Your choices for quasi-Newton acceleration are none (0),
         BFGS ('bfgs'), or SR-1 ('sr1'). The default as of today is BFGS, but
         I may change that as I have done in the past.

  random_stencil: Set this to k > 0 and k random unit vectors will be
         added to the stencil before each call to the simplex gradient.
         The default is 0.

  scale_step, _start, _depth: The range of scales is scale_step^-scale_start
         to scale_step^-scale_depth. The defaults are scale_step=2,
         scale_start=3, scale_depth=11.

  scale_aware: Setting this to 1 tells imfil that f can do something smart if
         it knows the scale. The call then becomes [fout,ifail,icount]=f(x,h)
         The scale is the second input argument. The default is 0 (not
         scale-aware). You can set scale_aware=1 and noise_aware=1 at the same
         time.

  simple_function: Set this to 1 and imfil will not ask you for either a cost
         estimate or a failure flag. Instead imfil will assume all costs are
         the same (and set them to 1) and that your function never, EVER,
         fails. Your function can then return only the value (or values if
         parallel=1). So the calling sequence is

            fout=f(x,h) (where h is optional, see scale_aware)

         The default is 0 (your function returns three things).

  smooth_problem: Set this to 1 and you will configure imfil to solve smooth
         problems. This option is equivalent to

                  bscales=[.5, .01, .001, .0001, .00001];
                    options=optset(
                        custom_scales=bscales,
                        stencil_wins='yes',
                        limit_quasi_newton='off',
                        armijo_reduction=0.25,
                        maxitarm=5)

         If you use this option you should probably increase your budget and
         experiment with both the default BFGS quasi-Newton method and SR1.

  stencil: imfil has three built-in stencils, with room for you to create you
        own in the matrix options. Setting stencil lets you chose the centered
        difference stencil (0), the forward difference stencil (1), and the
        positive basis stencil (2). The default is the centered difference
        stencil (stencil = 0). Choosing the forward difference stencil will
        cause a warning, as that option will go away in a future version.

  stencil_wins: Setting this to 'yes' or 1 will make the new point the better
        of the best point in the stencil and the point returned by the line
        search. The default 'no' is a bias in favor of the quasi-Newton point,
        and sets the point to the quasi-Newton point if the line search
        succeeds, and to the best point in the stencil only if the line search
        fails.

  target: You may have either a good lower estimate of the optimal value of f,
        or an idea about when the optimization has done enough. If you do, set
        target to that value and the optimization will stop when f < target.

  termtol: The inner iteration will terminate successfully if
        fscale*norm(proj(difference_grad)) < termtol * h.
        The default is termtol = .01.
        The inner iteration usually terminates with stencil failure, so there
        is little reason to change termtol unless you have a smooth problem,
        and then you might want to reduce it. Increasing termtol will reduce
        the number of total calls to f, but may also lead to premature
        termination and very little progress.

  svarmin: Set svarmin if you have a good estimate of the noise.
        If svarmin > 0 then imfil will declare stencil failure if the
        difference between the max and min values on the stencil is < svarmin.
        This option is the same as setting noise_aware to 1 and letting the
        function return svarmin as the noise_level. If the noise is not a
        function of h, setting svarmin once is easier.


  Matrix options

  custom_scales: You may set you own scales with an array
        H=[h(1), ... h(smax)] where 1 > h(1) > .. > h(smax) > 0
        The syntax is optout=optset(optin, custom_scales=H)

  vstencil: You may replace the built-in stencils with your own custom
        design. If VS is your N x M list of M directions, then
            optout=optset(optin, vstencil=VS)
        will get it done.

  Advanced Options: WARNING!!! You have to be aware of imfil_core's internal
  scaling to use this stuff. RTFM. I am not able to help you with your use of
  these options. You are on your own. Please read the documentation and look
  at the examples.

  add_new_directions:
         This function, which you must provide, will examine the current
         stencil, scale, and history of the iteration and add directions to
         the stencil. You may want to do this, for example, to add tangent
         directions for nearly active linear constraints.

         You enable this with
           options=optset(options, add_new_directions=mystencil);
         where the calling sequence of your function must be

           Vadd=mystencil(x,h,V);

           where x = current iterate;
                 h = current scale;
                 V = stencil for this iteration (which may include any random
                     directions you asked for with the random_stencil option)

  explore_function:
         This function, which you write, will explore design space after each
         inner iteration is complete.

         Turn it on with
           options=optset(options, explore_function=my_search);
         where the calling sequence of your search function must be

           [xs, fs, complete_history] = my_search(complete_history)

         xs is the best point from your search, fs=f(xs), and you must update
         the complete_history array, so that array is passed back and forth to
         your function.

         Setting this option also sets explore to 1

  executive_function:
         You write an executive function to replace the internal quasi-Newton
         or Gauss-Newton solvers in imfil.

         Turn the executive_function on with
           options=optset(executive_function=my_solver,
                          executive_data=my_data)
         Your function is my_solver and the data you function needs is
         my_data. You must include my_data in the call even if you don't use
         it.

         The calling sequence for your function is
           [xp, fvalp, funp, fcost, iarm, solver_hist, nfail, new_data]
             = my_solver(f, x, fun, sdiff, xc,  gc, iteration_data, old_data)

         Please, PLEASE, read the manual before messing with the executive
         function.


  ----------------------------------------------------------------

  The extra_argument options are set for you when you add the extra
  argument to your call to imfil. There is no reason to set them
  with imfil_optset.

  extra_argument: 1 or 'yes' if you are passing data through imfil
                  to the function,
                  0 or 'no' other wise

  extra_arg_value: the data you are passing to the function if
                   'extra_argument' is on.

  ----------------------------------------------------------------

  Please look at the example for this in the software collection and in
  the users' guide."""

    if not optin:
    # Start from default structure
        optout = _create_defaults()
    else:
        optout = copy.copy(optin)

    for name, val in optkwds.items():
        optout = _optset_one(name, val, optout)

    return optout


#-----
def _create_defaults():
    """Default values"""

    class OptSet(object):
        __slots__ = {
            'add_new_directions' : [],  # function to add new directions.
            #
            'armijo_reduction' : .5,    # Step size reduction factor for Armijo rule.
            #
            'complete_history' : 0,     # Store complete history.
            #
            'custom_scales' : [],       # vector of custom scales
            #
            'executive' : 0,            # No executive.
            #
            'executive_function' : [],  # What function?
            #
            'executive_data' : [],      # What data?
            #
            'executive_data_flag' : 0,  # No data! Really!
            #
            'explore_data_flag' : 0,    # The exploration step needs no data.
            #
            'explore_data' : [],        # No data means no data.
            #
            'explore' : 0,              # no exploration step
            #
            'explore_function' : [],    # If I'm not exploring, why write a function?
            #
            'extra_argument' : 0,       # passing something to f?
            #
            'extra_arg_value' : [],     # what you pass to f if extra_argument = 1
            #
            'fscale' : -1.2,            # Scale function by 1.2 times abs(f(x_0)).
            #
            'function_delta' : -1,      # Do not terminate on small differences of successive
                                        # successful iterations.
            #
            'stencil_delta' : -1,       # Do not terminate the optimization 
                                        # on small total variation of f.
            #
            'least_squares' : 0,        # Is this a nonlinear least squares problem?
            #
            'limit_quasi_newton' : 1,   # Don't let the quasi-Newton direction get too long.
            #
            'maxit' : 50,               # At most maxit*n quasi-Newton iterations/scale.
            #
            'maxitarm' : 3,             # Limit on number of step length reductions.
            #
            'maxfail' : 3,              # Limit on consecutive line search/stencil failures.
            #
            'noise_aware' : 0,          # Can your function estimate the noise? Default = no.
            #
            'parallel' : 0,             # Serial is the default.
            #
            'quasi' : 1,                # BFGS
            #
            'random_stencil' : 0,       # Leave the stencil alone.
            #
            'scale_aware' : 0,          # Can your function use the scale?
            #
            'scale_step' : 2,           # Stepping factor for scales
            #
            'scale_start' : 1,          # Scales begin at h=(1/scale_step)^scale_start
            #
            'scale_depth' : 8,          # Scales end at h=(1/scale_step)^(scale_start+scale_depth)
            #
            'simple_function' : 0,      # Your function returns more than just the value.
            #
            'smooth_problem' : 0,       # Don't use imfil for smooth problems if you can
                                        # avoid it.
            #
            'standalone': STANDALONE,   # Use orginal interface if standalone, scikit-quant iface if not
            #
            'stencil' : 0,              # Central difference stencil.
            #
            'stencil_wins' : 0,         # Who's the new point?
            #
            'target' : -1.E8,           # Target for optimal value = -1.d8.
                                        # You should try to do better.
            #
            'termtol' : .01,            # Small enough to let stencil failure do its job.
            #
            'svarmin' : 0,              # I have no idea what the noise is. 
            #
            'verbose' : 0,              # Shut up.
            #
            'vstencil' : [],            # No fancy stencil.
        }

        def __init__(self):
            for key, value in self.__slots__.items():
                setattr(self, key, value)

    return OptSet()


#-----
def _optset_one(name, val, optin):
    """
  This is the internal function to manage the options structure."""

    name = name.lower();

    # Is this an advanced option?
    if name in ('add_new_directions', 'extra_arg_value',
                'explore_function', 'executive_function'):
        adv_flag = 1
    else:
        adv_flag = 0

    # Test for verbal shortcuts and do the right thing.
    # The advanced options are set to names of functions and 
    # do not use verbal shortcuts.
    if isinstance(val, str) and adv_flag == 0:
        optin = verbal_shortcut(name, val, optin)
    else:
        # Update options.
        setattr(optin, name, val)

    # Now put in the flags for the explore_function and executive options.
    if name == 'explore_function':
        optin.explore = 1
    elif name == 'explore_data':
        optin.explore_data_flag = 1
    elif name == 'explore':
        yesno = optin.explore
        if not yesno:
            optin.explore_function  = []
            optin.explore_data_flag = 0
            optinexplore_data       = []
    elif name == 'executive_function':
        optin.executive = 1
    elif name == 'executive_data':
        optin.executive_data_flag = 1
    elif name == 'executive':
        yesno = getfield(optin, 'explore');
        if not yesno:
            optin.executive_function  = []
            optin.executive_data_flag = 0
            optin.executive_data      = []

    return optin


#-----
def verbal_shortcut(name, val, optin):
    """
  Plugs into imfil_optset and lets you use verbal shortcuts
  to control options with yes/no, on/off, or short lists of
  options."""

    val = val.lower();
    if name == 'add_new_directions':
       return optin
    elif name == 'complete_history':
        valout = yes_no(val)
    elif name == 'explore':
        valout = yes_no(val)
    elif name == 'least_squares':
        valout = yes_no(val)
    elif name == 'limit_quasi_newton':
        valout = yes_no(val)
    elif name == 'quasi':
        if val == 'bfgs':
            valout = 1
        elif val == 'sr1':
            valout = 2
        elif val == 'none':
            valout = 0
        else:
            raise ValueError('unexpected value for %s:' % name, val)
    elif name == 'scale_aware':
        valout = yes_no(val)
    elif name == 'simple_function':
        valout = yes_no(val)
    elif name == 'smooth_problem':
        valout = yes_no(val)
    elif name == 'stencil_wins':
        valout = yes_no(val)
    elif name == 'verbose':
        valout = yes_no(val)
    else:
        raise ValueError('no verbal shortcut found for %s' % str(name))

    setattr(optin, name, valout)
    return optin


#-----
def yes_no(val):
    if val in ['yes', 'on']:
        return True
    if val in ['no', 'off']:
        return False
    raise ValueError(val)
