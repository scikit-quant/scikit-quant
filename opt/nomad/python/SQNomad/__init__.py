import logging

log = logging.getLogger('SKQ.Nomad')
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
C. Audet, S. Le Digabel, C. Tribes and V. Rochon Montplaisir. "The NOMAD
project." Software available at https://www.gerad.ca/nomad .
S. Le Digabel. "NOMAD: Nonlinear Optimization with the MADS algorithm."
ACM Trans. on Mathematical Software, 37(4):44:1–44:15, 2011.
------------------------------------------------------------------------""")

__all__ = ['minimize', 'log']

from ._nomad import minimize
