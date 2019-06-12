from __future__ import print_function
import logging

log = logging.getLogger('SQSnobFit')
log.addHandler(logging.StreamHandler())
if log.level == logging.NOTSET:
    log.setLevel(logging.INFO)

log.info("""------------------------------------------------------------------------
W. Huyer and A. Neumaier, "Snobfit - Stable Noisy Optimization by Branch and Fit",
 ACM Trans. Math. Software 35 (2008), Article 9.
Software available at www.mat.univie.ac.at/~neum/software/snobfit
------------------------------------------------------------------------""")

__all__ = ['minimize', 'optset', 'log', 'snobfit']

from ._snobfit import minimize, snobfit
from ._optset import optset
