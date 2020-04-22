import logging

log = logging.getLogger('SKQ.ImFil')
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
C.T. Kelley, "Implicit Filtering", 2011, ISBN: 978-1-61197-189-7
Software available at ctk.math.ncsu.edu/imfil.html
------------------------------------------------------------------------""")

__all__ = ['minimize', 'optset', 'log']

from ._imfil import minimize
from ._optset import optset
