from __future__ import print_function

from SQCommon import Result, ObjectiveFunction


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
        import _pybobyqa as optimizer

    if optimizer is not None:
        return optimizer.minimize(func, x0, bounds, budget, options, **optkwds)

    raise RuntimeError('unknown optimizer "%s"' % method)
