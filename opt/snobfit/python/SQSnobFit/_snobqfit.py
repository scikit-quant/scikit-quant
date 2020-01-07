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
 function y, f1 = snobqfit(j, x, f, near, dx, u, v)

 A quadratic model around the best point is fitted and minimized with
 minq over a trust region.
 
 Input:
  j            index of the best point
  x            the rows contain the points where the function has been
               evaluated
  f            corresponding function values, i.e., f[i] = f[x(i]]
  near         near[i] is a vector containing the indices of the nearest
               neighbors of the point x[i]
  dx           resolution vector, i.e. the ith coordinate of a point to be
               generated is an integer-valued multiple of dx[i]
  u, v         the points are to be generated in [u, v]
 
 Output:
  y            minimizer of the quadratic model around the best point
  f1           its estimated function value
"""

from ._gen_utils import rand, sort
from ._snoblocf import snobround
from .minq import minq
import numpy


def snobqfit(j, x, f, near, dx, u, v):
    n = x.shape[1]	# dimension of the problem
    K = min(len(x)-1, n*(n+3))
    nneigh = near.shape[1]
    x0 = x[j]
    f0 = f[j]
    distance = numpy.sum((x - numpy.outer(numpy.ones(len(x)), x0))**2, 1)
    dd, ind = sort(distance)
    ind = ind[1:K+1]
    nind = len(ind)
    d = numpy.abs(x[near[j]]-numpy.outer(numpy.ones(nneigh), x0)).max(0)
    d = numpy.maximum(d, dx)
    S = x[ind] - numpy.outer(numpy.ones(K), x0)
    R = numpy.triu(numpy.linalg.qr(S, 'r'))
    R = R[:n]      # unnecessary?
    L = numpy.linalg.inv(R).T
    sc = numpy.sum((S.dot(L.T))**2, 1)**(3/2.0)
    b = (f[ind]-f0)/sc
    A = numpy.concatenate((x[ind,:] - numpy.outer(numpy.ones(K), x0),
                        0.5*(x[ind,:] - numpy.outer(numpy.ones(K), x0))**2), 1)

    for i in range(n-1):
        B = (x[ind,i]-x0[i]).reshape(nind,1).dot(numpy.ones((1,n-i-1)))
        A = numpy.concatenate((A, B*(x[ind, i+1:n]-(numpy.ones((K, 1)).dot(x0[i+1:n][None,:])))), axis=1)

    A = A/(sc.reshape(nind,1).dot(numpy.ones((1, A.shape[1]))))
    y = numpy.linalg.lstsq(A, b, rcond=None)[0]

    G = numpy.zeros((n, n))
    for i in range(n):
        G[i,i] = y[n+i]

    l = 2*n
    for i in range(n-1):
        for j in range(i+1,n):
            G[i,j] = y[l]
            G[j,i] = y[l]
            l += 1

    g = y[:n]
    c = g - G.dot(x0.T).reshape(u.shape)
    dp = d.reshape(u.shape); x0p = x0.reshape(u.shape)
    y, f1, ierr, nsub = minq(f0-x0.dot(g) + 0.5*x0.dot(G.dot(x0.T)), c, G,
            numpy.maximum(x0p.T - dp.T, u.T), numpy.minimum(x0p.T + dp.T, v.T), 0)
    y = snobround(y.T, u, v, dx)
    nc = 0
    while (numpy.min(numpy.max(numpy.abs(x-numpy.outer(numpy.ones(len(x)), y))
            - numpy.outer(numpy.ones(len(x)), dx), axis=1)) < -numpy.spacing(1)) and nc < 10:
        u1 = numpy.maximum(x0 - d, u)
        v1 = numpy.minimum(x0 + d, v)
        y = u1 + rand(1,n)*(v1-u1)
        y = snobround(y, u, v, dx)
        nc += 1

    f1 = f0+(y-x0).dot(g) + 0.5*((y-x0).dot(G.dot((y-x0).T)))

    return y.flatten(), float(f1)
