# Copyright UC Regents

from SQCommon import Result, ObjectiveFunction

import libsqnomad


def minimize(f, x0, bounds=None, budget=100, options=None, **kwds):
    opts = dict()
    if options:
       opts.update(options)
    if kwds:
       opts.update(kwds)

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

    if bounds:
        lower = bounds[:,0].flatten()
        upper = bounds[:,1].flatten()
    else:
        lower, upper = None, None

    fbest, xbest = libsqnomad.minimize(
        objfunc, x0, lower, upper, **opts)

    return Result(fbest, xbest), objfunc.get_history()
