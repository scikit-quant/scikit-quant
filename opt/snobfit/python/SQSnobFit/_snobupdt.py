from __future__ import print_function
# Python version of SNOBFIT v2.1 "snobfit.m" MATLAB version by A. Neumaier.
#
# Modified and redistributed with permission.

# Original copyright and license notice: #
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
# [iNCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# [iNCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
 function xl, xu, x, f, nsplit, small, near, d, np, t, inew, fnan, u, v = \
   snobupdt(xl, xu, x, f, nsplit, small, near, d, np, t, xnew, fnew, fnan, u, v, u1, v1, dx)

 Updates the box parameters when a set of new points and their function
 values are added; i.e. the boxes containing more than one point are
 split and the nearest neighbors are computed or updated.

 Input:
  xl, xu        rows contain lower and upper bounds of the old boxes
  x             rows contain the old points
  f             f[j] contains the function value at x[j], its
                variation and other parameters
  nsplit        nsplit[j,i] number of times box j has been split along
                the ith coordinate
  small         small[j] is an integer-valued logarithmic volume measure
                of box j
  near          near[j] is a vector pointing to the nearest
                neighbors of x[j]
  d             d[j] is the maximal distance between x[j] and one of
                its neighbors
  np            np[j] is the number of times the function value of
                x[j] has been measured
  t             t[j] is np[j] times the variance of the function values
                measured for the point x[j]
  xnew          rows contain new points
  fnew          new function values and their variations
                fnew[j,0] = f(xnew[j]), fnew[j,1] = df(xnew[j])
  fnan          pointer to all old points where the function value could
                not be obtained
  u, v          box bounds
  u1, v1        box in which the new points are to be generated
  dx            resolution vector

 Output:
  xl, xu       updated version of xl, xu [including new boxes)
  x            updated version of x
  f            updated version of f
  nsplit       updated version of nsplit
  small        updated version of small
  near         updated version of near
  d            updated version of d
  np           updated version of np
  t            updated version of t
  inew         pointer pointing to the new boxes and boxes whose
               nearest neighbors have changed
  fnan         possibly updated version of fnan [if a function value
               was found for a point in the new iteration)
  u, v         possibly updated box bounds such that all new points
               are in the box
"""

from ._gen_utils import find, min_, rsort, sort
from ._snobinput import snobinput
from ._snobnn    import snobnn
from ._snobsplit import snobsplit
import numpy


def snobupdt(xl, xu, x, f, nsplit, small, near, d, np, t, xnew, fnew, fnan, u, v, u1, v1, dx):
    n = u.shape[1]      # dimension of the problem
    nneigh = n+5
    nxold = len(x)      # number of points from the previous iteration
    nxnew = len(xnew)

    inew = numpy.array([], dtype=int)
    if x.size > 0:
      # if any of the new points are already among the old points, they are
      # thrown away and the function value and its uncertainty are updated
        dismiss = []
        for j in range(nxnew):
            i = find(sum(abs(numpy.ones((nxold,1))*xnew[j]-x),1) == 0)
            if i.size > 0:
                if not find(fnan==i).any() and numpy.isfinite(f[i,1]): # point i had finite
                                                                       # function value
                    if not numpy.isnan(fnew[j,0]):
                        np[i] = np[i] + 1
                        delta = fnew[j,0] - f[i,0]
                        f[i,0] = f[i,0] + delta/np[i]
                        t[i] = t[i] + delta*(fnew[j,0]-f[i,0])
                        f[i,1] = math.sqrt(f[i,1]**2 + (delta*(fnew[j,0] - f[i,0]) \
                            + fnew[j,1]**2 - f[i,1]**2) / np[i])
                        inew = numpy.concatenate((inew, i))

                    dismiss = numpy.concatenate((dismiss, [j]))
                else: # point i had NaN function value
                    if not numpy.isnan(fnew[j,0]).any():
                        f[i,0] = fnew[j,0]
                        inew = numpy.concatenate((inew, i))
                        ii = find(fnan==i)
                        if ii.size > 0:
                            fnan[ii] = numpy.array([], dtype=int)

                    dismiss = numpy.concatenate((dismiss, [j]))

        xnew = numpy.delete(xnew, dismiss, 0)
        fnew = numpy.delete(fnew, dismiss, 0)
        nxnew = len(xnew)
        if not nxnew:
            inew = numpy.sort(inew)
            return xl, xu, x, f, nsplit, small, near, d, np, t, inew.astype(int), fnan, u, v

    xnew, fnew, npnew, tnew = snobinput(xnew, fnew)
    nxnew = xnew.shape[0]

    nx = nxold + nxnew	# current number of points

    if numpy.sum(numpy.vstack((xnew, u)).min(0) < u) or \
        numpy.sum(numpy.vstack((xnew, v)).max(0) > v) or \
        (numpy.minimum(u, u1) < u).any() or (numpy.maximum(v, v1) > v).any():
        xl, xu, small, u, v = snobnewb(xnew, xl, xu, small, u, v, u1, v1)

    if x.size > 0:
        x = numpy.concatenate((x, xnew))
    else:
        x = xnew.copy()
    inew = numpy.concatenate((inew, numpy.arange(nxold, nx)), 0)
    if f.size <= 0:
        f = fnew.copy()
    else:
        if fnew.shape[1] < f.shape[1]:
            fnew = numpy.append(fnew, numpy.zeros((len(fnew), f.shape[1]-fnew.shape[1])), axis=1)
        f = numpy.vstack((f, fnew))
    if np.size > 0:
        np = numpy.append(np, npnew)
    else:
        np = npnew.copy()
    if t.size > 0:
        t = numpy.append(t, tnew)
    else:
        t = tnew.copy()
    if not nxold:
        xl, xu, x, f, nsplit, small = snobsplit(x, f, u, v, None, u, v)
    else:
        par = numpy.zeros((nxnew,))
        for j in range(nxnew):
            xx = numpy.ones((nxold,1))*xnew[j]
            ind = find(numpy.sum( \
                numpy.logical_and(numpy.less_equal(xl, xx), numpy.less_equal(xx, xu)), 1) == n)
            if ind.size > 0:
                minsmall, ismall = min_(small[ind])
                par[j] = ind[ismall]

        par1, ww, cdfx, dof = rsort(par)
        inew = numpy.concatenate((inew, par1), 0)
        for j in par1.astype(int):
            ind = find(par==j)
            ind = ind + nxold
            spl = numpy.append([j], ind.flatten())
            xl0, xu0, x0, f0, nsplit0, small0 = \
                snobsplit(x[spl], f[spl], xl[j], xu[j], nsplit[j], u, v)
            nxj = len(ind) + 1    # number of points in box [xl[j],xu[j]]
            k = find(numpy.sum(x0 == numpy.ones((nxj,1))*x[j,:], axis=1) == n)
            xl[j] = xl0[k]
            xu[j] = xu0[k]
            nsplit[j] = nsplit0[k]
            small[j] = small0[k]
            for k in range(nxj-1):
                k1 = ind[k]
                k2 = find(numpy.sum(x0 == numpy.ones((nxj,1))*x[k1,:],axis=1) == n)
                if len(k1) > 1:
                    msmall, k3 = min_(small[k1])
                    k2 = k2[k3]
                ik1 = int(k1)
                if len(xl) <= ik1:
                    xl = numpy.append(xl, numpy.zeros((ik1-len(xl)+1, xl.shape[1])), axis=0)
                if len(xu) <= ik1:
                    xu = numpy.append(xu, numpy.zeros((ik1-len(xu)+1, xu.shape[1])), axis=0)
                if len(nsplit) <= ik1:
                    nsplit = numpy.append(nsplit, numpy.zeros((ik1-len(nsplit)+1, nsplit.shape[1])), axis=0)
                if len(small) <= ik1:
                    small = numpy.append(small, numpy.zeros((ik1-len(small)+1,)), axis=0)
                xl[k1] = xl0[k2]
                xu[k1] = xu0[k2]
                nsplit[k1] = nsplit0[k2]
                small[k1] = small0[k2]

    notnan = numpy.arange(0, nx)
    notnan = numpy.delete(notnan, fnan)
    notnan = numpy.delete(notnan, find(numpy.isnan(f[:,0])))

    if notnan.size > 0:
        fmn = numpy.min(f[notnan,0])
        fmx = numpy.max(f[notnan,0])
    else:
        fmn = 1
        fmx = 0

    if nx >= nneigh and fmn < fmx:
        if near.size <= 0:
            near = numpy.zeros((nx, nneigh), dtype=int)
        if d.size <= 0:
            d = numpy.zeros((nx,))

        for j in range(nxold, nx):
            jnear, jd = snobnn(x[j], x, nneigh, dx)
            if len(near) <= j:
                near = numpy.append(near, numpy.zeros((j-len(near)+1, nneigh), dtype=int), axis=0)
            near[j] = jnear

            if len(d) <= j:
                d = numpy.append(d, numpy.zeros((j-len(d)+1,)))
            d[j] = jd

        for j in range(nxold):
            if numpy.min(numpy.sqrt(numpy.sum((numpy.ones((nxnew,1))*x[j]-xnew)**2,1))) < d[j]:
                jnear, jd = snobnn(x[j], x, nneigh, dx)
                if len(near) <= j:
                    near = numpy.append(near, numpy.zeros((j-len(near)+1, nneigh)), axis=0)
                near[j] = jnear

                if len(d) <= j:
                    d = numpy.append(d, numpy.zeros((j-len(d)+1,)))
                d[j] = jd
                inew = numpy.concatenate((inew, [j]))

        inew = sort(inew)[0]
        d = d.reshape((1, len(d)))
    else:
        near = numpy.array([])
        d = numpy.inf*numpy.ones((1, nx))

    return xl, xu, x, f, nsplit, small, near, d, np, t, inew.astype(int), fnan, u, v


def snobnewb(xnew, xl, xu, small, u, v, u1, v1):
    nx = len(xl)
    n = nx and xl.shape[1] or 0
    uold = u
    vold = v
    u = numpy.concatenate((xnew, u)).min(0)
    v = numpy.concatenate((xnew, v)).max(0)
    u = numpy.min(u, u1)
    v = numpy.max(v, v1)
    i1 = find(u < uold)
    i2 = find(v > vold)
    ind = numpy.array([])
    for j in range(len(i1)):
        j1 = find(xl[:,i1[j]] == uold[i1[j]])
        ind = numpy.concatenate((ind, j1))
        xl[j1,i1[j]] = u[i1[j]]

    for j in range(len(i2)):
        j2 = find(xu[:, i2[j]] == vold[i2[j]])
        ind = numpy.concatenate((ind, j2))
        xu[j2,i2[j]] = v[i2[j]]

    if len(i1) + len(i2):    # at least one of the bounds was changed
        small = -numpy.sum(numpy.round(numpy.log2((xu-xl)/(numpy.ones((nx,1))*(v-u)))), axis=1)

    return xl, xu, small, u, v
