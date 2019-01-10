from __future__ import print_function

print("""------------------------------------------------------------------------
C.T. Kelley, "Implicit Filtering", 2011, ISBN: 978-1-61197-189-7
Software available at https://ctk.math.ncsu.edu/imfil.html
------------------------------------------------------------------------""")

__all__ = ['optimize', 'optset']

from ._imfil import *
from ._optset import *
