from __future__ import print_function

__all__ = ['ObjectiveFunction']

from ._stats import Stats
import numpy


class ObjectiveFunction(object):
    def __init__(self, func, options = {}):
        self.simple_function = False

        self.objective = func
        self.stats = Stats()
        try:
            for key, value in options.items():
                setattr(self, key, value)
        except:
            pass   # options is not a dict

    def __call__(self, par):
        self.stats.nevals += 1

        result = self.objective(par)
        try:
            nres = len(result)
        except TypeError:
            nres = 0

        err_estimate = 0.; cost = 1
        if nres == 0:
            val = result
        else:
            val = result[0]
            if nres >= 2:
                err_estimate = result[1]
            if nres >= 3:
                cost = result[2]
        self.stats.add_history(val, par)

        if self.simple_function:
            return val
        return result

    def get_history(self):
        return self.stats.full_history()
