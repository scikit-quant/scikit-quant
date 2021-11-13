# Copyright UC Regents

from SQCommon import Result, ObjectiveFunction

import libsqnomad
import numpy as np


def minimize(f, x0, bounds=None, budget=100, options=None, **kwds):
    opts = dict()
    if options:
       opts.update(options)
    if kwds:
       opts.update(kwds)

  # ensure the use of numpy arrays
    if x0 is not None and not isinstance(x0, np.ndarray):
        x0 = np.array(x0)

  # TODO: processing of unknown keyworks is all string-based; should parse
  # the .txt files in src/Attributes instead and take up their types
    for key, value in opts.items():
        tt = type(value)
        if tt == bool:
            opts[key] = value and "true" or "false"
        elif type(value) != str:
            opts[key] = str(value)

  # add budget
    opts['MAX_BB_EVAL'] = budget

    objfunc = ObjectiveFunction(f)

    if bounds is not None:
        if not isinstance(bounds, np.ndarray):
            bounds = np.array(bounds)
        lower = bounds[:,0].flatten()
        upper = bounds[:,1].flatten()
    else:
        lower, upper = None, None

    fbest, xbest = libsqnomad.minimize(
        objfunc, x0, lower, upper, **opts)

    return Result(fbest, xbest), objfunc.get_history()
