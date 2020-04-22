import numpy

__all__ = ['CompleteHistory', 'append_history', 'single_point_hist_update',
    'many_point_hist_update', 'scan_history', 'scan_complete_history']


#-----
class CompleteHistory(object):
    def __init__(self, good_points=[], good_values=[], failed_points=[]):
        self.good_points   = list(good_points)
        self.good_values   = list(good_values)
        self.failed_points = list(failed_points)

    def copy(self):
        return CompleteHistory(
            self.good_points[:], self.good_values[:], self.failed_points[:])

    def __str__(self):
        return "good_points = %s\n good_values = %s\n failed_points = %s\n" %\
            (str(self.good_points), str(self.good_values), str(self.failed_points))


#-----
def append_history(histout, fcount, fval, npgrad, stepn, iarm, x):
    try:
        if len(fval) == 1:
            fvalhist = float(fval)
    except TypeError:
        fvalhist = fval
    histout.append([fcount, fvalhist, npgrad, stepn, iarm] + list(map(float, x)))


#-----
def single_point_hist_update(hist, x, fout, ifail):
    """
  Update the complete_history structure after a single call to f.
 
  single_point_hist_update(hist, x, fout, ifail)"""

    # Write the data.
    if ifail == 1:
        hist.failed_points.append(x)
    else:
        hist.good_points.append(x)
        hist.good_values.append(fout)


#-----
def many_point_hist_update(hist, diff_hist, inplace=True):
    """
  many_point_hist_update(hist, diff_hist)
 
  Update the complete_history structure after a many calls to f.
  
  WARNING! The complete_history structure uses your bounds, and is
  not scaled to make 0 <= x(i) <= 1."""

    if not inplace:
        hist = hist.copy()
    if type(diff_hist.good_points) == list:
        hist.good_points.extend(diff_hist.good_points)
    else:
        for i in range(diff_hist.good_points.shape[1]):
            hist.good_points.append(diff_hist.good_points[:,i])
    hist.good_values.extend(diff_hist.good_values)
    hist.failed_points.extend(diff_hist.failed_points)
    return hist


#-----
def scan_history(complete_history, xp, fp, dx):
    """
  Find previous evaluations, if any, at the requested points <xp>."""

    oldresults = {}
    newpoints = {}
    for i in range(xp.shape[1]):
        point = xp[:,(i,)]
        try:
            fpt, ift = scan_complete_history(complete_history, point)
            oldresults[i] = (fpt, ift)
        except LookupError:
            newpoints[i] = point

    return oldresults, newpoints


#-----
def scan_complete_history(complete_history, x):
    """
  Search for a previously evaluated point within 1.E-12 of <x>,
  returns previous value and associated flag."""

    for i, point in enumerate(complete_history.good_points):
        d = numpy.linalg.norm(x-point, ord=numpy.inf)
        if d < 1.E-12:
            return complete_history.good_values[:,i], 0

    for i, point in enumerate(complete_history.failed_points):
        d = numpy.linalg.norm(x-point, ord=numpy.inf)
        if d < 1.E-12:
            return numpy.NaN, 1

    raise LookupError("no such point")
