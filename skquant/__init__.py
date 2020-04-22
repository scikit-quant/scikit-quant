"""Integrated software for quantum computing.

Submodules:
    opt: black-box optimers for noisy problems.

See http://scikit-quant.org for complete documentation.
"""

import logging


__author__ = 'Wim Lavrijsen <WLavrijsen@lbl.gov>'

__all__ = [
    'opt',              # black-box optimers for noisy problems
    'log',              # global logger
]


log = logging.getLogger('SKQ')
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
