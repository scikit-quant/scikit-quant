def minimize(func, x0, bounds, budget=10000, method='imfil', optin=None, **optkwds):
    method_ = method.lower()
    if method_ == 'imfil':
        import SQImFil
        return SQImFil.minimize(func, x0, bounds, budget, optin, **optkwds)

    raise RuntimeError('unknown optimizer "%s"' % method)
