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
  function x, fct, ier, nsub = minq(gam, c, G, xu, xo, prt, xx)

  Minimizes an affine quadratic form subject to simple bounds:
  using coordinate searches and reduced subspace minimizations
  using LDL^T factorization updates
     min    fct =  gam + c^T x + 0.5 x^T G x
     s.t.   x in [xu:xo]    # xu< = xo is assumed
  where G is symmetric n x n
  (if G is indefinite: only a local minimum is found).

  Output:
  prt           print level
  xx            initial guess (optional)
  x             minimizer (but unbounded direction if ier = 1)
  fct           optimal function value
  ier            0 (local minimizer found)
                 1 (unbounded below)
                99 (maxit exceeded)
"""

from ._gen_utils import diag, find
from .minq_subroutines import getalp, ldldown, ldlup
import copy, logging, math, numpy

logger = logging.getLogger('SKQ.SnobFit.minq')


def minq(gam, c, G, xu, xo, prt, xx=None):
    prt = 0

    c = c.reshape(xu.shape)
    convex = 0
    n = len(G)
    hpeps = 100 * numpy.spacing(1)  # perturbation in last two digits
    maxit = 3 * n      # maximal number of iterations
                       # this limits the work to about 1+4*maxit/n matrix multiplies
                       # usually at most 2*n iterations are needed for convergence
    nitrefmax = 3      # maximal number of iterative refinement steps

  # initialize trial point xx: function value fct and gradient g
    if xx is None:
      # cold start with absolutely smallest feasible point
        xx = numpy.zeros(n).reshape(xu.shape)

  # force starting point into the box
    xx = numpy.maximum(xu, numpy.minimum(xx, xo))

  # initialize factorization
    K = numpy.zeros(n, dtype=int) # initially no rows in factorization
    L = numpy.eye(n)
    dd = numpy.ones(n)            # LDL^T factorization of G_KK
    #if issparse('G'):
    #    print('better: use symbolic factorization (not yet implemented)')

  # dummy initialization of indicator of free variables
  # will become correct after first coordinate search
    free = numpy.zeros(n, dtype=int)
    nfree = 0
    nfree_old = -1

    fct = numpy.inf               # best function value
    nsub = 0                      # number of subspace steps
    unfix = 1                     # allow variables to be freed in csearch?
    nitref = 0                    # no iterative refinement steps so far
    improvement = True            # improvement expected

  ########################################################################
  # main loop: alternating coordinate and subspace searches
    while True:
        logger.debug('enter main loop')

        if numpy.linalg.norm(xx, numpy.inf) == numpy.inf:
            error('infinite xx in minq.m')

        g = G.dot(xx) + c
        fctnew = float(gam + 0.5*xx.T.dot(c+g))
        if not improvement:
          # good termination
            logger.debug('terminate: no improvement in coordinate search')
            ier = 0
            break

        elif nitref > nitrefmax:
          # good termination
            logger.debug('terminate: nitref>nitrefmax')
            ier = 0
            break

        elif nitref > 0 and nfree_old == nfree and fctnew >= fct:
          # good termination
            logger.debug('terminate: nitref > 0 and nfree_old == nfree and fctnew >= fct')

            ier = 0
            break

        elif nitref == 0:
            x = xx[:]
            fct = min(fct, fctnew)
            logger.debug('fct: %s', fct)

        else:  # more accurate g and hence f if nitref>0
            x = xx[:]
            fct = fctnew
            logger.debug('fct: %s', fct)

        if nitref == 0 and nsub >= maxit:
            logger.debug('incomplete minimization (too many iterations): increase maxit')
            logger.info('iteration limit exceeded')
            ier = 99
            break

        ######################################################################
        # coordinate search
        count = 0        # number of consecutive free steps
        k = -1           # current coordinate searched
        while True:
            while count <= n:
              # find next free index (or next index if unfix)
                count += 1
                if k == n-1:
                    k = -1
                k += 1
                if free[k] or unfix:
                    break

            if count > n:
              # complete sweep performed without fixing a new active bound
                break

            q = G[:,k]
            alpu = xu[k] - x[k]
            alpo = xo[k] - x[k]      # bounds on step

          # find step size
            alp, lba, uba, ier = getalp(alpu, alpo, float(g[k]), float(q[k]))
            if ier:
                x = numpy.zeros(n)
                if lba:
                    x[k] = -1
                else:
                    x[k] = 1

                if logger.getEffectiveLevel() >= logging.DEBUG:
                    logger.debug('function unbounded below, unbounded direction returned')
                    logger.debug('possibly caused by roundoff')
                    logger.debug('f(alp*x) = gam+gam1*alp+gam2*alp^2/2: where')
                    logger.debug('gam1 = %s', (c.T).dot(x))
                    logger.debug('gam2 = %s', (x.T).dot(G.dot(x)))
                    logger.debug('ddd = %s', diag(G))
                    logger.debug('min_diag_G = %s', ddd.min())
                    logger.debug('max_diag_G = %s', ddd.max())

                return x, fct, ier, nsub       

            xnew = x[k] + alp
            if nitref > 0:
                logger.debug('xnew: %s alp: %s', xnew, alp)

            if lba or xnew <= xu[k]:
              # lower bound active
                logger.debug('%d at lower bound', k)
                if alpu != 0:
                    x[k] = xu[k]
                    g = g + (alpu*q).reshape(g.shape)
                    count = 0
                free[k] = 0
            elif uba or xnew >= xo[k]:
                # upper bound active
                logger.debug( '%d at upper bound', k)
                if alpo != 0:
                    x[k] = xo[k]
                    g = g + (alpo*q).reshape(g.shape)
                    count = 0
                free[k] = 0
            else:
                # no bound active
                logger.debug('%d free', k)
                if numpy.spacing(1) < abs(alp):
                    if not free[k]:
                        logger.debug('unfixstep: %s %s', x[k], alp)

                    x[k] = xnew
                    g = g + (alp*q).reshape(g.shape)
                    free[k] = 1

            # end of coordinate search

        ######################################################################
        nfree = int(numpy.sum(free))
        if unfix and nfree_old == nfree:
          # in exact arithmetic: we are already optimal
          # recompute gradient for iterative refinement
            g = G.dot(x) + c
            nitref += 1
            logger.debug('optimum found; iterative refinement tried')

        else:
            nitref = 0

        nfree_old = nfree
        g = g.reshape(c.shape)
        gain_cs = float(fct - gam -0.5*x.T.dot(c+g))
        improvement = gain_cs > 10*numpy.spacing(1) or not unfix

      # print (0,1) profile of free and return the number of nonnp.zeros
        #nfree = pr01('csrch ', free)
        logger.debug('gain_cs: %s', gain_cs)

      # subspace search
        xx = x[:]
        if not improvement or nitref >= nitrefmax:
          # optimal point found or enough refinement steps - nothing done
            pass
        elif nfree == 0:
            pass
          # no free variables - no subspace step taken
            logger.debug('no free variables - no subspace step taken')
            unfix = 1
        else:
          # take a subspace step
            nsub += 1

            fct_cs = gam+0.5*x.T.dot(c+g.reshape(c.shape))
            format='*** nsub = %4.0f fct = %15.6e fct_cs = %15.6e'
            logger.debug(format % (nsub, fct, fct_cs))

          # downdate factorization
            for j in find(free < K):    # list of newly active indices
                L, dd = ldldown(L, dd, int(j))
                K[j] = 0
                if logger.getEffectiveLevel() >= logging.DEBUG:
                    logger.debug('downdate; fact_ind: %s', find(K).flatten())

          # update factorization or find indefinite search direction
            definite = 1
            for j in find(free > K):   # list of newly freed indices
              # later: speed up the following by passing K to ldlup!
                p = numpy.zeros(n)
                if n > 1:
                    idk = find(K)
                    for kk in idk:
                        p[kk] = G[kk, j]
                p[j] = G[j, j]
                L, dd, p = ldlup(L, dd, int(j), p)
                definite = p.size <= 0
                if not definite:
                    logger.debug('indefinite or illconditioned step')
                    break
                K[j] = 1
                if logger.getEffectiveLevel() >= logging.DEBUG:
                    logger.debug('update; fact_ind %s', find(K).flatten())

            if definite:
              # find reduced Newton direction
                p = numpy.zeros(n)
                idk = find(K).flatten()
                for kk in idk:
                    p[kk] = g[kk]
                p = -numpy.linalg.solve(L.T, numpy.linalg.solve(L, p)/dd)
                # p will remain zero and ignored
                if logger.getEffectiveLevel() >= logging.DEBUG:
                    logger.debug('reduced Newton step; fact_ind: %s', find(K).flatten())

          # set tiny entries to zero
          # p = (x+p)-x
            for ii in range(len(p)):
                if abs(p[ii]) < 100*numpy.spacing(1):
                    p[ii] = 0.0
            ind = find(p != 0).flatten()
            if ind.size <= 0:
              # zero direction
                logger.debug('zero direction')
                unfix = 1
                continue

          # find range of step sizes
            pp = p[ind].reshape(x[ind].shape)
            oo = (xo[ind]-x[ind])/pp
            uu = (xu[ind]-x[ind])/pp
            alpu = -numpy.inf 
            if oo[pp<0].size > 0: alpu = max(alpu, numpy.max(oo[pp<0]))
            if uu[pp>0].size > 0: alpu = max(alpu, numpy.max(uu[pp>0]))
            alpo =  numpy.inf
            if oo[pp>0].size > 0: alpo = min(alpo, numpy.min(oo[pp>0]))
            if uu[pp<0].size > 0: alpo = min(alpo, numpy.min(uu[pp<0]))
          # TODO: original had <= and =>, and alpo == 0.0 happens
            if alpo < 0 or alpu > 0:
                logger.debug("current alpo, alpu: %f, %f", alpo, alpu)
                raise RuntimeError('programming error: no alp')

          # find step size
            gTp = g.T.dot(p)
            agTp = numpy.abs(g).T.dot(numpy.abs(p))
            if abs(gTp) < 100*numpy.spacing(1)*agTp:
              # linear term consists of roundoff only
                gTp = 0
            pTGp = p.T.dot(G.dot(p))
            if convex: pTGp = max(0, pTGp)
            if not definite and pTGp > 0:
                logger.debug('tiny pTGp = %s set to zero', pTGp)
                pTGp = 0

            alp, lba, uba, ier = getalp(alpu, alpo, gTp, pTGp)
            if ier:
                x = numpy.zeros(n)
                if lba: x = -p
                else: x = p
                if logger.getEffectiveLevel() >= logging.DEBUG:
                    qg = gTp/agTp
                    qG = pTGp/(numpy.linalg.norm(p, 1)**2*numpy.linalg.norm(G[:], numpy.inf))
                    lam = numpy.linalg.eig(G)
                    lam1 = numpy.min(lam)/numpy.max(abs(lam))
                    logger.debug("minq: function unbounded below\n"
                                 "  unbounded subspace direction returned\n"
                                 "  possibly caused by roundoff\n"
                                 "  regularize G to avoid this!")

                    logger.debug('f(alp*x)=gam+gam1*alp+gam2*alp^2/2, where')
                    logger.debug('gam1 = %s', c.T.dot(x))
                    logger.debug('rel1 = %s', gam1/(numpy.abs(c).T.dot(numpy.abs(x))))
                    gam2 = x.T.dot(G.dot(x))
                    if convex: gam2 = max(0, gam2)
                    logger.debug('gam2 = %s', gam2)
                    logger.debug('rel2 = %s', gam2/(numpy.abs(x).T.dot((numpy.abs(G).dot(numpy.abs(x))))))
                    logger.debug('ddd = %s', diag(G))
                    logger.debug('min_diag_G = %s', min(ddd))
                    logger.debug('max_diag_G = %s', max(ddd))

                return x, fct, ier, nsub

            unfix = not (lba or uba)   # allow variables to be freed in csearch?
            
          # update of xx
            for k in range(len(ind)):
              # avoid roundoff for active bounds
                ik = int(ind[k])
                if alp == uu[k]:
                    xx[ik] = xu[ik]
                    free[ik] = 0
                elif alp == oo[k]:
                    xx[ik] = xo[ik]
                    free[ik] = 0
                else:
                    logger.debug('generic update')
                    xx[ik] = xx[ik]+alp*p[ik]

                if numpy.abs(xx[ik]) == numpy.inf:
                    logger.debug("%s %s %s", ik, alp, p[ik])
                    raise RuntimeError('infinite xx in minq.py')

            nfree = sum(free)

      # print (0:1) profile of free and return the number of nonzeros
        #nfree = pr01('ssrch ', free)
        if unfix and numpy.sum(nfree) < n:
            logger.debug('bounds may be freed in next csearch')


  # end of main loop
    logger.debug('fct: %s', fct)

    return x, fct, ier, nsub
