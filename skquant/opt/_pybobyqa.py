from __future__ import print_function

from SQCommon import Result, ObjectiveFunction
import pybobyqa


def minimize(func, x0, bounds, budget, optin, **optkwds):
     objfunc = ObjectiveFunction(func)

     # massage bounds
     lower = bounds[:,0]
     upper = bounds[:,1]

     # actual Py-BOBYQA call
     result = pybobyqa.solve(
        objfunc, x0, maxfun=budget, bounds=(lower,upper), objfun_has_noise=True, **optkwds)

     # get collected history and repackage return result
     full_history = objfunc.get_history()
     return Result(result.f, result.x), full_history, full_history
