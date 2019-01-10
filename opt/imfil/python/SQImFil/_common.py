from __future__ import print_function

__all__ = ['Result', 'Stats']


class Result(object):
    def __init__(self, val, par):
        self.optval = val
        self.optpar = par

class Stats(object):
    def __init__(self, nevals):
        self.nevals = nevals
