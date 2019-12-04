from __future__ import print_function

# additional features to provide:
#   - logging in pybobyqa
#   - seek_global_minimum flag for global bias

from SQCommon import Result, ObjectiveFunction
import logging, numpy, pybobyqa

log = logging.getLogger('SKQ.PyBobyqa')

log.info("""
------------------------------------------------------------------------
Coralia Cartis, et. al., "Improving the Flexibility and Robustness of
 Model-Based Derivative-Free Optimization Solvers", technical report,
 University of Oxford, (2018).
Software available at github.com/numericalalgorithmsgroup/pybobyqa/
------------------------------------------------------------------------""")

__all__ = ['minimize', 'log']


def minimize(func, x0, bounds, budget, optin, **optkwds):
     objfunc = ObjectiveFunction(func, {'simple_function' : True })

     # massage bounds (force reshaping as bobyqa is picky)
     lower = numpy.asarray(bounds[:,0]).reshape(-1)
     upper = numpy.asarray(bounds[:,1]).reshape(-1)

     x0 = numpy.asarray(x0).reshape(-1)

     # actual Py-BOBYQA call
     result = pybobyqa.solve(
        objfunc, x0, maxfun=budget, bounds=(lower,upper), seek_global_minimum=True, objfun_has_noise=True, **optkwds)

     # get collected history and repackage return result
     return Result(result.f, result.x), objfunc.get_history()
