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
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES
# LOSS OF USE, DATA, OR PROFITS OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
 function xl, xu, x, f, nsplit, small = snobsplit(x, f, xl0, xu0, nspl, u, v)

 Splits a box [xl0,xu0] contained in a bigger box [u,v] such that each of the
 resulting boxes contains just one point of a given set of points.

 Input:
  x             the rows are a set of points
  f - f[j,:]    contains the function value, its variation and possibly
                other parameters belonging to point x(j,:)
  xl0           vector of lower bounds of the box
  xu0           vector of upper bounds of the box
  nspl          nspl[i] is the number of splits the box [xl0,xu0] has already
                undergone along the ith coordinate
  u, v          bounds of the original problem; [xl0, xu0] is contained in [u, v]

 Output:
  xl - xl[j,:]  is the lower bound of box j
  xu - xi[j,:]  is the upper bound of box j
  x - x[j,:]    is the point contained in box j
  f - f[j,:]    contains the function value at x(j,:), its uncertainty etc.
  nsplit - nsplit[j,i] is the number of times box j has been split in the
                ith coordinate
  small - small[j] integer-valued logarithmic volume measure of box j
"""

from ._gen_utils import rsort, max_, find, std
import numpy


def snobsplit(x, f, xl0, xu0, nspl=None, u=None, v=None):
    n = x.shape[1]
    if len(xl0) > 1:
        xl0 = xl0.T
        xu0 = xu0.T

    if nspl is None or nspl.size <= 0:
        nspl = numpy.zeros((1, n))

    if u is None or u.size <= 0:
        u = xl0.copy()
    if v is None or v.size <= 0:
        v = xu0.copy()

    if len(x) == 1:
        xl = xl0.copy()
        xu = xu0.copy()
        nsplit = nspl.copy()
        small = numpy.array([-numpy.sum(numpy.round(numpy.log2((xu-xl)/(v-u))))])
        return xl, xu, x, f, nsplit, small

    elif len(x) == 2:
        dmax, i = max_((numpy.abs(x[0]-x[1])/(v-u)).flatten())
        ymid = 0.5*(x[0,i]+x[1,i])
        xl = numpy.zeros((2, n))
        xu = numpy.zeros((2, n))
        xl[0] = xl0
        xu[0] = xu0
        xl[1] = xl0
        xu[1] = xu0
        if x[0,i] < x[1,i]:
            xu[0,i] = ymid
            xl[1,i] = ymid
        else:
            xl[0,i] = ymid
            xu[1,i] = ymid

        nsplit = numpy.zeros((2, n))
        nsplit[0] = nspl
        nsplit[0,i] = nsplit[0,i] + 1
        nsplit[1,i] = nsplit[0,i]
        small = -numpy.sum(numpy.round(numpy.log2((xu-xl)/(numpy.ones((2,1))*(v-u)))),1).T
        return xl, xu, x, f, nsplit, small

    var = numpy.zeros(v.shape[1])

    for i in range(n):
        var[i] = std(x[:,i]/(v.T[i]-u.T[i]))

    xx = numpy.sort(x,0)
    dd = xx[1:xx.shape[0]]-xx[0:xx.shape[0]-1]
    mvar, i = max_(var)
    y, w, cdf, dof = rsort(x[:,i], remove_dups=False)

    d = y[1:y.shape[0]] - y[0:y.shape[0]-1]
    ld = len(d)

    ii = range(ld)
    dmax, j = max_(d[ii])
    j = ii[j]

    ymid = 0.5*(y[j] + y[j+1])
    ind1 = find(x[:,i] < ymid)
    ind2 = find(x[:,i] > ymid)

    xl = numpy.zeros((2, n))
    xu = numpy.zeros((2, n))
    nsplit = numpy.zeros((2, n))
    npoint = numpy.zeros(2)
    ind = numpy.zeros((2, max(len(ind1), len(ind2))), dtype=int)
    xl[0] = xl0
    xu[0] = xu0
    xu[0,i] = ymid
    xl[1] = xl0
    xl[1,i] = ymid
    xu[1] = xu0
    nsplit[0] = nspl
    nsplit[0,i] = nsplit[0,i] + 1
    nsplit[1] = nsplit[1,:]
    npoint[0] = len(ind1)
    npoint[1] = len(ind2)
    # Note: adding 1 here, so that later we can search for 0 as "False"
    ind[0,:len(ind1)] = ind1.T+1
    ind[1,:len(ind2)] = ind2.T+1
    nboxes = 1

    maxpoint, j = max_(npoint)
    while maxpoint > 1: 
        ind0 = ind[j,find(ind[j,:])].astype(int)
        # Undo the add of 1 to make these values back to indices again
        ind0 -= 1
        for i_ in range(n):
            while i_ >= len(var):
                var = numpy.append(var, 0)
            var[i_] = std(x[ind0,i_]/(v.T[i_]-u.T[i_]))

        maxvar, i = max_(var)
        rsort_x = (x[ind0,i])[:,0]
        y, w, cdf, dof = rsort(rsort_x, remove_dups=False)

        d = y[1:len(y)] - y[0:len(y)-1]
        ld = len(d)

        ii = numpy.arange(ld).astype(int)
        dmax, k = max_(d[ii])
        k = ii[k]
        ymid = 0.5*(y[k]+y[k+1])
        ind1 = find(x[ind0,i]<ymid)
        ind2 = find(x[ind0,i]>ymid)
        ind1 = ind0[ind1[:,0],0]
        ind2 = ind0[ind2[:,0],0]
        nboxes = nboxes + 1

        if len(xl) <= nboxes:
            xl = numpy.append(xl, numpy.zeros((len(xl)-nboxes+1, xl.shape[1])), axis=0)
        if len(xu) <= nboxes:
            xu = numpy.append(xu, numpy.zeros((len(xu)-nboxes+1, xu.shape[1])), axis=0)
        if len(nsplit) <= nboxes:
            nsplit = numpy.append(nsplit, numpy.zeros((len(nsplit)-nboxes+1, nsplit.shape[1])), axis=0)

        while nboxes >= len(npoint):
            npoint = numpy.append(npoint, 0)
        while nboxes >= ind.shape[0]:
            ind = numpy.append(ind, [numpy.zeros(ind.shape[1])], axis=0)

        xl[nboxes] = xl[j]
        xu[nboxes] = xu[j]
        xu[j,i] = ymid
        xl[nboxes,i] = ymid
        nsplit[j,i] = nsplit[j,i] + 1
        nsplit[nboxes,:] = nsplit[j,:]
        npoint[j] = len(ind1)
        npoint[nboxes] = len(ind2)
        ind[j,:ind.shape[1]] = 0
        # Note: here, too, adding 1, so that later we can search for 0 as "False"
        ind[j,:len(ind1)] = ind1+1
        ind[nboxes,:len(ind2)] = ind2+1
        if not numpy.sum(ind[:,ind.shape[1] - 1]):
            ind = ind[:,:ind.shape[1] - 1]
        maxpoint, j = max_(npoint)

    if ind.size <= 0:
        x = numpy.array([None])
        f = numpy.array([None])
    else:
        ind = ind.astype(int)
        x = x[ind[:,0]-1]
        f = f[ind[:,0]-1]
    small = -numpy.sum(numpy.round(numpy.log2((xu-xl)/(numpy.ones((x.shape[0],1))*(v-u)))),1).T

    return xl, xu, x, f, nsplit, small
