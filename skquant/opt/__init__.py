from __future__ import print_function

import subprocess
import sys
from SQCommon import Result, ObjectiveFunction


__all__ = [
    'methods',
    'minimize',
    ]

def _check_orbit_prerequisites():
    try:
        import rpy2
        return True
    except ImportError:
        pass
    return False

def methods():
    """Returns a list of available optimizer methods"""

    m = ['bobyqa', 'imfil']
    if _check_orbit_prerequisites():
        m.append('orbit')
    m.append('snobfit')
    return m


def minimize(func, x0, bounds, budget=10000, method='imfil', options=None, **optkwds):
    optimizer = None

    method_ = method.lower()
    if 'imfil' in method_:
        import SQImFil as optimizer
        import SQImFil._optset as _optset   # ImFil standalone has a different API
        _optset.STANDALONE = False
    elif 'snobfit' in method_ :
        import SQSnobFit as optimizer
    elif 'bobyqa' in method_:
        import skquant.opt._pybobyqa as optimizer
    elif 'orbit' in method_:
        if not _check_orbit_prerequisites():
            raise RuntimeError("ORBIT is only supported on Python3 because of rpy2")
        import skquant.opt._norbitR as optimizer

    if optimizer is not None:
        return optimizer.minimize(func, x0, bounds, budget, options, **optkwds)

    raise RuntimeError('unknown optimizer "%s"' % method)
