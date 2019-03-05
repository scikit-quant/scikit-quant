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
  function x1 = snob5(x, u, v, dx, nreq)
  generates nreq points of class 5 in [u,v]
 
  Input:
  x       the rows are the points already chosen by Snobfit (can be
          empty)
  u, v    bounds of the box in which the points are to be generated
  dx      resolution vector, i.e. the ith coordinate of a point to be
          generated is an integer-valued multiple of dx(i)
  nreq    number of points to be generated
 
  Output:
  x1      the rows are the requested points
          x1 is of dimension nreq x n (n = dimension of the problem)
"""
  
from ._gen_utils import find, max_, rand
from ._snoblocf  import snobround
import numpy


def snob5(x, u, v, dx, nreq):
    n = len(u)     # dimension of the problem
    nx = len(x)    # number of data points
    nx1 = 100*nreq
    d = numpy.zeros(nx1, dtype=int)
    xnew = numpy.outer(numpy.ones(nx1), u) + rand(nx1,n)*numpy.outer(numpy.ones(nx1), v-u)
    if nx:
        for j in range(nx1):
            xnew[j,:] = snobround(xnew[j,:], u, v, dx)
            d[j] = numpy.min(numpy.sum((x-numpy.ones((nx,1))*xnew[j,:])**2, 1))

        ind = find(d == 0)
        xnew = numpy.delete(xnew, ind, 0)
        d = numpy.delete(d, ind, 0)
        x1 = numpy.array([])
        nx1 = xnew.shape[0]
    else:
        x1 = xnew[0,:]
        xnew = xnew[1:]
        nx1 = nx1-1
        d = numpy.sum((xnew - numpy.outer(numpy.ones(nx1), x1))**2, axis=1, keepdims=True).flatten()
        nreq = nreq - 1

    for j in range(nreq):
        if d.size <= 0:
            break
        dmax, i = max_(d)
        y = xnew[i]
        if x1.size > 0:
            x1 = numpy.vstack((x1, y))
        else:
            x1 = y.copy()
        xnew = numpy.delete(xnew, i, 0)
        d = numpy.delete(d, i, 0)
        nx1 = nx1 - 1
        d1 = numpy.sum((xnew - numpy.outer(numpy.ones(nx1), y))**2, 1)
        d = numpy.minimum(d, d1)

    return x1
