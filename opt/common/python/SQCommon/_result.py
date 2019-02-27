from __future__ import print_function

__all__ = ['Result']


class Result(object):
    def __init__(self, val, par):
        self.optval = val
        self.optpar = par

    def __str__(self):
        return "Optimal value: %g, at parameters: %s" % (self.optval, str(self.optpar))
