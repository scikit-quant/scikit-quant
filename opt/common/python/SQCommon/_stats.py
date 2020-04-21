from __future__ import print_function

import numpy

__all__ = ['Result']


class Stats(object):
    def __init__(self, nevals = 0):
        self.nevals = nevals
        self._history = list()

    def add_history(self, fval, par):
        self._history.append([fval]+list(numpy.atleast_1d(par)))

    def full_history(self):
        """Return the full call history. Each row represents one call, with
           the first column the result from the objective function call, the
           other columns the parameters used."""

        return numpy.array(self._history)
