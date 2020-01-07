from __future__ import print_function
# Python version of SNOBFIT v2.1 "snobinput.m" MATLAB version by A. Neumaier.
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
 function x, f, np, t = snobinput(x, f)

 Checks whether there are any duplicates among the points given by the rows of
 x, throws away the duplicates and computes their average function values and an
 estimated uncertainty.

 Input:
  x            numpy array of points
  f            numpy array, f[j,0] is the function value of x[j] and f[j,1] is its
               uncertainty

 Output: (x, f, np, t)
  x            updated version of x (possibly some points have been deleted)
  f            updated version of f (f[j,0] is the average function value and
               f[j,1] the estimated uncertainty pertaining to x[j])
  np           np[j] is the number of times the row x[j] appeared in the
               input version of x
  t - t[j] is np[j] times the variance of the function values measured
               for point x[j]
"""

import math, numpy


def snobinput(x, f):
    if x.size <= 0:
        return x, f, numpy.array([]), numpy.array([])
    sx = len(x)                   # num points
    n = sx and x.shape[1] or 0    # num parameters per point
    i = 0

    nullp = numpy.ones(sx)
    t = numpy.ones(sx)

    while i < sx:
        j = i + 1
        ind = numpy.array([])

        while j < sx:
            if numpy.array_equal(x[i],x[j]):
                ind = numpy.concatenate((ind,[j]))
            j += 1

        if (len(ind) != 0):
            ind = numpy.concatenate(([i],ind)).astype(int)
            ind1 = numpy.transpose(numpy.nonzero(numpy.isnan(f[ind,0]))).astype(int)

            if len(ind1) < len(ind):
                ind = numpy.delete(ind, ind1).astype(int)
                nullp[i] = len(ind)

                fbar = numpy.sum(f[ind,0])/nullp[i]
                t[i] = numpy.sum((f[ind,0]-fbar)**2)
                f[i,0] = fbar
                f_dev = f[i,1]
                if(not isinstance(f[i,1], numpy.float64) ):
                    f_dev = sum(f[i,1])

                f[i,1] = math.sqrt((f_dev**2+t[i])/nullp[i])
            else:
                nullp[i] = 1
                t[i] = 0

            x = numpy.delete(x, ind[1:], 0)
            f = numpy.delete(f, ind[1:], 0)
            sx = x.shape[0]
        else:
            nullp[i] = 1
            t[i] = 0

        i += 1

    return x, f, nullp, t
