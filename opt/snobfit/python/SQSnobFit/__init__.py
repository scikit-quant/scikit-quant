from __future__ import print_function
import logging

log = logging.getLogger('SKQ.SnobFit')
if not log.hasHandlers():
    def _setupLogger(log):
        import sys
        hdlr = logging.StreamHandler(sys.stdout)
        frmt = logging.Formatter('%(name)-12s: %(levelname)8s %(message)s')
        hdlr.setFormatter(frmt)
        log.addHandler(hdlr)
        log.propagate = False
    _setupLogger(log)
    del _setupLogger


log.info("""
------------------------------------------------------------------------
W. Huyer and A. Neumaier, "Snobfit - Stable Noisy Optimization by Branch and Fit",
 ACM Trans. Math. Software 35 (2008), Article 9.
Software available at www.mat.univie.ac.at/~neum/software/snobfit
------------------------------------------------------------------------""")

__all__ = ['minimize', 'optset', 'log', 'snobfit']

from ._snobfit import minimize, snobfit
from ._optset import optset
