# Copyright UC Regents

from SQCommon import Result, ObjectiveFunction

import libsqnomad


def minimize(f, x0, bounds, budget=100, optin=None, **optkwds):
    opts = {
        'MAX_BB_EVAL' : budget,
    }

    objfunc = ObjectiveFunction(f)

    fbest, xbest = libsqnomad.minimize(objfunc, x0,
        bounds[:,0].flatten(), bounds[:,1].flatten(), **opts)

    return Result(fbest, xbest), objfunc.get_history()
