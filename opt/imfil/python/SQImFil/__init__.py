from __future__ import print_function

### temporary override of NumPy deprecation warning, see:
### https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html
import warnings
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

import logging

log = logging.getLogger('SQImFil')
log.addHandler(logging.StreamHandler())
if log.level == logging.NOTSET:
    log.setLevel(logging.INFO)

log.info("""------------------------------------------------------------------------
C.T. Kelley, "Implicit Filtering", 2011, ISBN: 978-1-61197-189-7
Software available at ctk.math.ncsu.edu/imfil.html
------------------------------------------------------------------------""")

__all__ = ['optimize', 'optset']

from ._imfil import optimize
from ._optset import optset
