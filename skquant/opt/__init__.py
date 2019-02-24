def minimize(func, x0, bounds, budget=10000, method='imfil', optin=None, **optkwds):
    optimizer = None

    method_ = method.lower()
    if method_ == 'imfil':
        import SQImFil as optimizer
    elif method_ == 'snobfit':
        import SQSnobFit as optimizer

    if optimizer is not None:
        return optimizer.minimize(func, x0, bounds, budget, optin, **optkwds)

    raise RuntimeError('unknown optimizer "%s"' % method)
