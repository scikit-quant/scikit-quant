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
#       names of its contributors may be used to endorse or promote products
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
 request, xbest, fbest = snobfit(x, f, config, dx=None)
 minimization of a function over a box in R^n

 Input:
 file          name of file for input and output
               if nargin < 5, the program continues a previous run and
               reads from file.mat the output is (again) stored in file.mat

^^do not use file - store variables globally,
or make them available to be passed in?

 x             the rows are a set of new points entering the
               optimization algorithm together with their function
               values
 f             matrix containing the corresponding function values
               and their uncertainties, i.e., f(j,1) = f(x(j))
               and f(j,2) = df(x(j))
               a value f(j,2)<=0 indicates that the corresponding
               uncertainty is not known, and the program resets it to
               sqrt(numpy.spacing(1))
 config        structure variable defining the box [u,v] in which the
               points are to be generated, the number nreq of
               points to be generated and the probability p that a
               point of type 4 is generated
               config = struct('bounds',{u,v},'nreq',nreq,'p',p)
 dx            only used for the definition of a new problem (when
               the program should continue from the values stored in
               file.mat, the call should have only 4 input parameters!)
               n-vector (n = dimension of the problem) of minimal
               stnp.spacing(1), i.e., two points are considered to be different
               if they differ by at least dx(i) in at least one
               coordinate i

 Output:
 request       nreq x (n+3)-matrix
               request(j,1:n) is the jth newly generated point,
               request(j,n+1) is its estimated function value and
               request(j,n+3) indicates for which reason the point
               request(j,1:n) has been generated
               request(j,n+3) = 1 best prediction
                              = 2 putative local minimizer
                              = 3 alternative good point
                              = 4 explore empty region
                              = 5 to fill up the required number of
                              function values if too little points of
                              the other classes are found
 xbest         current best point
 fbest         current best function value (i.e. function value at xbest)
"""

from SQCommon import Result, ObjectiveFunction
from ._gen_utils import diag, max_, min_, find, extend, rand, sort
from ._optset    import optset
from ._snobinput import snobinput
from ._snoblocf  import snoblocf, snobround
from ._snoblp    import snoblp
from ._snobnan   import snobnan
from ._snobpoint import snobpoint
from ._snobqfit  import snobqfit
from ._snobsplit import snobsplit
from ._snobupdt  import snobupdt
from ._snob5     import snob5
import logging, math, numpy

__all__ = ['minimize', 'log']

log = logging.getLogger('SKQ.SnobFit')


# dummy save/load (can always pickle but seems superflous; deal with this
# for full history later)
_im_storage = None
def _snobsave(*args):
    global _im_storage
    _im_storage = args

def _snobload():
    global _im_storage
    return _im_storage


'''
iter - number of iterations to run snobfit loop
f - use numpy array
x - use numpy multidimensional array
config - use dictionary
    bounds - multidimensional numpy array
    nreq - int
    probability p - float
'''
def snobfit_func(iter, x, func, config, dx, df):
    req = x
    f = numpy.zeros((len(x), 2))

    for n in iter:
        for ind in len(req):
            f[ind][0] = func(req[ind])
            f[ind][1] = df

        req, xbest, fbest = snobfit(req, f, config, dx)

    return req, xbest, fbest


def fill_request(request, func, nparams):
    x = numpy.zeros((len(request), nparams))
    f = numpy.zeros((len(request), 2))
    for i in range(len(request)):
        x[i] = request[i][0:nparams]
        try:
            res = func(x[i])
            err = math.sqrt(numpy.spacing(1))
            if type(res) == tuple:
                if res[1] != 0:
                    err = res[1]
                res = res[0]
            f[i] = (res, err)
        except Exception as e:
            log.Error('Function evaluation failed: %s', str(e))
            f[i] = numpy.nan
    return x, f


#-----
def minimize(f, x0, bounds, budget, optin={}, **optkwds):
    # The user-facing API is the equivalent of snobdriver, providing the loop
    # over the "internal" snobfit function.
    if budget <= 0:
        budget = 100000

    if type(x0) != numpy.ndarray:
        x0 = numpy.array(x0)

    if len(x0.shape) == 1:
        x0 = x0.reshape(1, len(x0))

    if type(bounds) != numpy.ndarray:
        bounds = numpy.array(bounds)

    objfunc = ObjectiveFunction(f)

    minfcall = 10;      # minimum number of function values before
                        # considering stopping

    # calculate resolution vector from the bounds
    dx = (bounds[:,1]-bounds[:,0])*1E-5

    # setup parameters (TODO: use optin/optkwds)
    if type(optin) == dict:
        options = optset(**dict(optin, **optkwds))
    else:
        options = optset(optin, **optkwds)

    config = {"bounds": bounds, "nreq": 2*len(bounds)+6, "p": .5}
    if optin is not None:
        if options.maxmp is not None:
            config["nreq"] = options.maxmp

    nstop = options.maxfail      # number of times no improvement is tolerated

    nparams = len(bounds)

    # initial call with empty list
    request, xbest, fbest = snobfit(
        numpy.array([]).reshape(0, len(bounds)), numpy.array([]).reshape(0,2), config, dx)

    # initial call with just x0 as input point(s) (establishes initial request)
    #request, xbest, fbest = snobfit(x0, numpy.array([]).reshape(0,2), config, dx)
    if options.verbose:
        print('request =', request)

    # calculate the requested points and set uncertainties
    x, vals = fill_request(request, objfunc, nparams)

    ncall0 = len(vals)                 # initial budget used
    fbestn, jbest = min_(vals[:,0])    # best function value
    xbest = x[jbest,:]

    # display current number of function values, best point and function value
    log.info('# calls = %d; xbest = %s; fbest = %f', ncall0, str(xbest), fbest)

    nstop0 = 0;
    # repeated calls to Snobfit
    while ncall0 < budget:   # repeat till ncall function values are reached
                             # (if the stopping criterion is not fulfilled first)
        request, xbest, fbest = snobfit(x, vals, config)
        if options.verbose:
            print('request =', request)

        # computation of the function values at the suggested points
        x, vals = fill_request(request, objfunc, nparams)

        # update function call counter
        ncall0 = ncall0 + len(vals)
        fbestn, jbest = min_(vals[:,0])    # best function value
        if fbestn < fbest:
            fbest = fbestn
            xbest = x[jbest,:]

            # display current number of function values
            log.info('# calls = %d; xbest = %s; fbest = %f', ncall0, str(xbest), fbest)

            nstop0 = 0
        elif budget >= minfcall:
            nstop0 = nstop0 + 1

        # check stopping criterion
        if nstop0 >= nstop and ncall0 >= minfcall:
            break

    return Result(fbest, xbest), objfunc.get_history()


def snobfit(x, f, config, dx = None):
    ind = find(f[:,1] <= 0)
    if not (ind.size <= 0 or numpy.all(ind==0)):
        f[ind,1] = math.sqrt(numpy.spacing(1))    # may be wrong

    rho = 0.5*(math.sqrt(5)-1)	  # golden section number
    u1 = config['bounds'][:,0]    # lower
    v1 = config['bounds'][:,1]    # upper

    nreq = config['nreq']
    p = config['p']
    n = len(u1)         # dimension of the problem
    nneigh = n+5        # number of nearest neighbors

    dy = 0.1*(v1-u1)    # defines the vector of minimal distances between two
                        # points suggested in a single call to Snobfit

    if dx is not None:  # a new job is started
        if numpy.any(dx<=0):
            raise ValueError('dx should contain only positive entries')
        if dx.shape[0] > 1:
            dx = dx.T
        if x.size > 0:
            u = numpy.minimum(x.min(axis=0), u1)
            v = numpy.maximum(x.max(axis=0), v1)
        else:
            u = u1[:]
            v = v1[:]

        x, f, np, t = snobinput(x, f)   # throw out duplicates among the points
                                        # and compute mean function value and
                                        # deviation
        if x.size > 0:
            xl, xu, x, f, nsplit, small = snobsplit(x, f, u, v, None, u, v)
            d = numpy.inf*numpy.ones((1, len(x)))
        else:
            xl = numpy.array([])
            xu = numpy.array([])
            nsplit = numpy.array([])
            small = numpy.array([])

        notnan = find(numpy.isfinite(f[:,0]))
        if notnan.size > 0:
            fmn = min_(f[notnan,1])
            fmx = max_(f[notnan,1])
        else:
            fmn = 1
            fmx = 0

        if (len(x) >= nneigh+1) and (fmn < fmx):
            inew = range(len(x))
            near = numpy.zeros((len(x), nneigh))
            d = numpy.zeros(len(x))
            for j in inew:
                near[j], d[j] = snobnn(x[j], x, nneigh, dx)

            fnan = find(numpy.isnan(f[:,0]))
            if fnan.size > 0:
                f = snobnan(fnan,f,near,inew)

            jsize = inew[-1]
            y = numpy.zeros((jsize, 2))
            g = numpy.zeros((jsize, 2))
            sigma = numpy.zeros(jsize)
            f = extend(f, 1)
            for j in inew:
                y[j], f[j, 2], c, sigma[j] = snoblocf(j, x, f[:,0:2], near, dx, u, v)
                g[j] = c.reshape(1, len(c))

            fbest, jbest = min_(f[:,0])
            xbest = x[jbest]
        else:
            fnan = numpy.array([], dtype=int)
            near = numpy.array([], dtype=int)
            d = numpy.inf*numpy.ones((1,len(x)))

            x1 = snob5(x, u1, v1, dx, nreq)
            request = numpy.concatenate((x1, numpy.nan*numpy.ones((nreq,1)), 5*numpy.ones((nreq,1))), 1)
            if x.size > 0 and f.size > 0:
                fbest, jbest = min_(f[:,0])
                xbest = x[jbest]
            else:
                xbest = numpy.nan*numpy.ones((1,n))
                fbest = numpy.inf

            if len(request) < nreq:
                snobwarn()

            y = None
            _snobsave(xbest, fbest, x, f, xl, xu, y, nsplit, small, near, d, np, t, fnan, u, v, dx)
            return request, xbest, fbest
    else:
        xnew = x.copy()
        fnew = f.copy()
        xbest, fbest, x, f, xl, xu, y, nsplit, small, near, d, np, t, fnan, u, v, dx = _snobload()
        nx = len(xnew)
        oldxbest = xbest

        xl, xu, x, f, nsplit, small, near, d, np, t, inew, fnan, u, v  = \
            snobupdt(xl, xu, x, f, nsplit, small, near, d, np, t, xnew, fnew, fnan, u, v, u1, v1, dx)

        if near.size > 0:
            ind = find(numpy.isnan(f[:,0]))
            if ind.size > 0:
                fnan = numpy.concatenate((fnan, ind.flatten()))
            if fnan.size > 0:
                f = snobnan(fnan, f, near, inew)

            fbest, jbest = min_(f[:,0])
            xbest = x[jbest]
            jsize = int(inew[-1]+1)
            if y is None:
                y = numpy.zeros((jsize, x.shape[1]))
            else:
                y = numpy.append(y, numpy.zeros((jsize-len(y), x.shape[1])), axis=0)
            g = numpy.zeros((jsize, x.shape[1]))
            sigma = numpy.zeros(jsize)
            f = extend(f, x.shape[1]-1)
            for j in inew:
                y[j], f[j,2], c, sigma[j] = snoblocf(j, x, f[:,0:2], near, dx, u, v)
                g[j] = c

        else:
            x1 = snob5(x, u1, v1, dx, nreq)
            request = numpy.concatenate((x1, numpy.NaN*numpy.ones((len(x1),1)), 5*numpy.ones((len(x1),1))), 1)
            if x.size > 0:
                (fbest, ibest) = min_(f[:,0])
                xbest = x[ibest]
            else:
                xbest = numpy.array([])
                fbest = numpy.inf

            if request.shape[0] < nreq:
                snobwarn()

            _snobsave(xbest, fbest, x, f, xl, xu, y, nsplit, small, near, d, np, t, fnan, u, v, dx)
            return request, xbest, fbest

    sx = len(x)
    request = numpy.array([]).reshape(0, x.shape[1]+2)
    ind = find(numpy.sum(numpy.logical_and(xl <= numpy.outer(numpy.ones(sx), v1), xu >= numpy.outer(numpy.ones(sx), u1)), 1) == n)
    minsmall, k = min_(small[ind])
    maxsmall = small[ind].max(0)
    m1 = numpy.floor((maxsmall-minsmall)/3)
    k = find(small[ind] == minsmall)
    k = ind[k].flatten()
    fsort, j = sort(f[k,0])
    k = k[j]
    isplit = k[0]

    if numpy.sum(numpy.logical_and(u1<=xbest, xbest<=v1)) == n:
        z, f1 = snobqfit(jbest, x, f[:,0], near, dx, u1, v1)
    else:
        fbes, jbes = min_(f[ind,0])
        jbes = ind[jbes]
        xbes = x[jbes]
        z, f1 = snobqfit(jbes, x, f[:,0], near, dx, u1, v1)

    z = snobround(z, u1, v1, dx)
    zz = numpy.outer(numpy.ones(sx), z)
    j = find(numpy.sum(numpy.logical_and(xl<=zz, zz<=xu), 1) == n)
    if len(j) > 1:
        msmall, j1 = min_(small[j])
        j = j[j1]

    if numpy.min(numpy.max(numpy.abs(x - numpy.outer(numpy.ones(sx), z)) - numpy.outer(numpy.ones(sx), dx))) >= -numpy.spacing(1):
        dmax = numpy.max((xu[j] - xl[j])/(v-u))
        dmin = numpy.min((xu[j] - xl[j])/(v-u))
        if dmin <= 0.05*dmax:
            isplit = numpy.append(isplit, j)
        else:
            request = numpy.vstack((request, numpy.concatenate((z, numpy.array((f1, 1))), 0)))

    if len(request) < nreq:
        globloc = nreq - len(request)
        glob1 = globloc*p
        glob2 = math.floor(glob1)
        if rand(1) < glob1 - glob2:
            glob = glob2 + 1
        else:
            glob = glob2

        loc = globloc - glob
        if loc:
            local, nlocal = snoblp(f[:,0], near, ind)
            fsort, k = sort(f[local,2]) #uhhhhhh
            j = 0
            sreq = len(request)
            while sreq < (nreq-glob) and j < len(local):
                l0 = local[k[j]]
                y1 = snobround(y[l0], u1, v1, dx)
                yy = numpy.outer(numpy.ones((len(x), 1)), y1)
                l = find(numpy.sum(numpy.logical_and(xl<=yy, yy<=xu), 1) == n)
                if len(l) > 1:
                    msmall, j1 = min_(small[l])
                    l = l[j1]

                dmax = numpy.max((xu[l] - xl[l]) / (v - u))
                dmin = numpy.min((xu[l] - xl[l]) / (v - u))
                if dmin <= 0.05*dmax:
                    isplit = numpy.append(isplit, l)
                    j += 1
                    continue

                if numpy.max(abs(y1-x[l])-dx) >= -numpy.spacing(1) and \
                        (not sreq or numpy.min( \
                             numpy.max(numpy.abs(request[:,0:n] - \
                             numpy.outer(numpy.ones(sreq), y1))-numpy.outer(numpy.ones(sreq), numpy.maximum(dy,dx)), axis=1)) >= -numpy.spacing(1)):
                    if numpy.sum(y1 == y[l0]) < n:
                        D = f[l0,1]/dx**2
                        #Possibly problem area!
                        f1 = f[l0,0] + g[l0].dot((y1 - x[l0]).T) + sigma[l0]*( \
                            (y1 - x[l0]).dot(diag(D).dot((y1-x[l0]).T) + f[l0,1]))
                    else:
                        f1 = f[l0,2]
                    request = numpy.vstack((request, numpy.concatenate((y1, numpy.array((f1, 2))), 0)))

                sreq = len(request)
                j += 1

            if sreq < nreq-glob:
                fsort, k = sort(f[nlocal,2])

            j = 0
            while sreq < (nreq-glob) and j < len(nlocal):
                l0 = nlocal[k[j]]
                y1 = snobround(y[l0], u1, v1, dx)
                yy = numpy.outer(numpy.ones(len(x)), y1)
                l = find(numpy.sum(numpy.logical_and(xl<=yy, yy<=xu),1) == n)
                if len(l) > 1:
                    msmall, j1 = min_(small[l])
                    l = l[j1]

                dmax = numpy.max((xu[l] - xl[l]) / (v - u))
                dmin = numpy.min((xu[l] - xl[l]) / (v - u))
                if dmin <= 0.05*dmax:
                    isplit = numpy.append(isplit, l)
                    j += 1
                    continue

                if numpy.max(numpy.abs(y1-x[l]) - dx) >= -numpy.spacing(1) and \
                        (not sreq or numpy.min( \
                             numpy.max(numpy.abs(request[:,:n] - numpy.outer(numpy.ones(sreq), y1)) - \
                             numpy.outer(numpy.ones(sreq), numpy.maximum(dy,dx)), axis=1)) >= -numpy.spacing(1)):
                    if numpy.sum(y1==y[l0]) < n:
                        D = f[l0,1]/(dx**2)
                        f1 = f[l0,0] + g[l0].dot((y1 - x[l0]).T) + \
                        sigma[l0]*(((y1 - x[l0]).dot(diag(D).dot((y1 - x[l0]).T))) + f[l0,1])
                    else:
                        f1 = f[l0,2]

                    request = numpy.vstack((request, numpy.concatenate((y1, numpy.array((f1, 3))), 0)))

                sreq = len(request)
                j += 1


    sreq = len(request)
    for l in isplit.flatten():
        jj = find(ind==l)
        ind = numpy.delete(ind, jj) #ind(jj) = []
        y1, f1 = snobpoint(x[l], xl[l], xu[l], f[l,0:2], g[l], sigma[l], u1, v1, dx)

        if numpy.max(numpy.abs(y1-x[l]) - dx) >= -numpy.spacing(1) and \
                (not sreq or numpy.min( \
                     numpy.max(numpy.abs(request[:,:n] - numpy.outer(numpy.ones(sreq), y1)) - \
                     numpy.outer(numpy.ones(sreq), dx), axis=1)) >= -numpy.spacing(1)):
            request = numpy.vstack((request, numpy.concatenate((y1, numpy.array((f1, 4))), 0)))

        sreq = len(request)
        if sreq == nreq:
            break

    first = True
    while (sreq < nreq) and ind.size > 0:   # and find(small[ind] <= (minsmall + m1)).any():
        for m in range(int(m1+1)):
            if first:
                first = False
                continue

            m = 0
            k = find(small[ind] == minsmall+m)
            while k.size <= 0:
                m += 1
                k = find(small[ind] == minsmall+m)

            if k.size > 0:
                k = ind[k].flatten()
                fsort, j = sort(f[k,0])
                k = k[j]
                l = int(k[0])
                jj = find(ind == l)
                ind = numpy.delete(ind, jj)
                y1, f1 = snobpoint(x[l], xl[l], xu[l], f[l,0:2], g[l], sigma[l], u1, v1, dx)
                if numpy.max(numpy.abs(y1-x[l]) - dx) >= -numpy.spacing(1) and \
                        (not sreq or numpy.min( \
                             numpy.max(numpy.abs(request[:,:n] - numpy.outer(numpy.ones(sreq), y1)) - \
                             numpy.outer(numpy.ones(sreq), numpy.maximum(dy,dx)), axis=1)) >= -numpy.spacing(1)):
                    request = numpy.vstack((request, numpy.concatenate((y1, numpy.array((f1, 4))), 0)))

                sreq = len(request)
                if sreq == nreq:
                    break
            m = 0

    if len(request) < nreq:
        x1 = snob5(numpy.concatenate((x, request[:,:n])), u1, v1, dx, nreq - len(request))
        nx = len(x)
        for j in range(len(x1)):
            i = find((numpy.sum(xl <= numpy.outer(numpy.ones(nx), x1[j])) and \
                   (numpy.outer(numpy.ones((nx,1)), x1[j]) <= xu), 1) == n)
            if len(i) > 1:
                minv, i1 = min_(small[i])
                i = i[i1]

            D = f[i,1]/(dx**2)
            f1 = f[i,0] + (x1[j] - x[i]).dot(g[i].T) + sigma[i]*((x1[j] - x[i]).dot(diag(D).dot((x1[j] - x[i]).T))) + f[i,1]
            request = numpy.vstack((request, numpy.concatenate((x1[j], numpy.array((f1, 5))), 0)))

    if len(request) < nreq:
        snobwarn()

    _snobsave(xbest, fbest, x, f, xl, xu, y, nsplit, small, near, d, np, t, fnan, u, v, dx)
    return request, xbest, fbest


def snobwarn():
    """issues a warning if SNOBFIT has not been able to generate the desired
       number of points"""

    import warnings
    warnings.warn("WARNING: The algorithm was not able to generate the desired number of points\n"
                  "Change the search region or refine dx")
