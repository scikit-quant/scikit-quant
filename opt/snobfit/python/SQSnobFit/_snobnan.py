from __future__ import print_function
# Python version of SNOBFIT v2.1 "snobfit.m" MATLAB version by A. Neumaier.
#
# Modified and redistributed with permission.

# Original copyright and license notice:
#
# Copyright (c) 2003-2008, Arnold Neumaier
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the University of Vienna nor the
#       names of its contributors may be used to orse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY ARNOLD NEUMAIER ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL  ARNOLD NEUMAIER BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
 function f = snobnan(fnan, f, near, inew)

 Replaces the function values NaN of a set of points by a value determined by
 their nearest neighbors with finite function values, with a safeguard for the
 case that all neighbors have function value NaN.
 
 Input:
  fnan         vector containing the pointers to the points where the
               function value could not be obtained
  f            f[:,0] set of available function values
               f[:,1] their uncertainty/variation
  near[j,:]    vector pointing to the nearest neighbors of point j
  inew         vector pointing to the new boxes and boxes whose nearest
               neighbors have changed
 
 Output:
  f            updated version of f
"""

from ._gen_utils import rsort, max_, min_, find, std
import numpy


def snobnan(fnan, f, near, inew):
    lnn = near.shape[1]
    fnan = fnan.flatten()
    notnan = list(filter(lambda x: x not in fnan, range(len(f))))
    fmx, imax = max_(f[notnan,0])
    fmn, imin = min_(f[notnan,0])
    dfmax = f[imax,1]

    for l in fnan:
        if find(inew==l).size > 0:
            # a substitute function value is only computed for new points and for
            # points whose function values have changed
            ind = near[l,:]

            # eliminate neighbors with function value NaN
            ind1 = numpy.array([], dtype=int)
            for i in range(len(ind)):
                 if (find(fnan == ind[i])).size > 0:
                     ind1 = numpy.concatenate((ind1, [i]), 0)

            if ind1.size > 0:
                ind = numpy.delete(ind, ind1, 0)

            if ind.size <= 0:
                f[l,0] = fmx + 1.e-3*(fmx-fmn)
                f[l,1] = dfmax
            else:
                 fmax1, k = max_(f[ind, 0])
                 fmin1, temp = min_(f[ind, 0])
                 f[l,0] = fmax1 + 1.e-3*(fmax1-fmin1)
                 f[l,1] = f[ind[k],1]

    return f
