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

from ._gen_utils import diag, find
import copy, numpy, logging

logger = logging.getLogger('SKQ.SnobFit.minq')


#-----
def getalp(alpu, alpo, gTp, pTGp):
    """
     function alp, lba, uba, ier = getalp(alpu, alpo, gTp, pTGp)

     Get minimizer alp in [alpu, alpo] for a univariate quadratic
            q(alp) = alp*gTp+0.5*alp^2*pTGp

     lba       lower bound active
     uba       upper bound active

     ier       0 (finite minimizer)
               1 (unbounded minimum)
    """

    lba = False
    uba = False

    # determine unboundedness
    ier = False
    if alpu == -numpy.inf and (pTGp < 0 or (pTGp == 0 and gTp > 0)):
        ier = True; lba = True

    if alpo ==  numpy.inf and (pTGp < 0 or (pTGp == 0 and gTp < 0)):
        ier = True; uba = True

    if ier:
        return numpy.nan, lba, uba, ier

    # determine activity
    if pTGp == 0 and gTp == 0:
        alp = 0
    elif pTGp <= 0:
        # concave case minimal at a bound
        if alpu == -numpy.inf:
            lba = False
        elif alpo == numpy.inf:
            lba = True
        else:
            lba = bool((2*gTp + (alpu+alpo)*pTGp) > 0)

        uba = not lba
    else:
        alp = -gTp/pTGp           # unconstrained optimal step
        lba = bool(alp <= alpu)   # lower bound active
        uba = bool(alp >= alpo)   # upper bound active

    if lba: alp = alpu
    if uba: alp = alpo

    # print?
    if abs(alp) == numpy.inf:
        print(gTp, pTGp, alpu, alpo, alp, lba, uba, ier)

    return alp, lba, uba, ier


#-----
def ldldown(L, d, j):
    """
      function L, d = ldldown(L, d, j)

      Downdates LDL^T factorization when j-th row and column are replaced
      by j-th unit vector.

      d contains diag(D) and is assumed positive
    """

    n = len(d)

    test = 0
    if test:
        logger.debug('enter ldldown')
        A = L.dot(diag(d).dot(L.T))
        A[:,j] = numpy.zeros(n)
        A[j] = numpy.zeros((1,n))
        A[j,j] = 1

    if j < n:
        I = numpy.arange(j)
        K = numpy.arange(j+1,n)
        L[K[:,None],K], d[K], p = ldlrk1(L[K[:,None],K], d[K], d[j], L[K,j])
      # update jth row and column
        L[j,I] = numpy.zeros((    j,))
        L[K,j] = numpy.zeros((n-1-j,))
    else:
        L[n-1,:n-2] = numpy.zeros((1,n-2))
    d[j] = 1

    if test:
        A1 = L.dot(diag(d)).dot(L.T)
        quot = numpy.linalg.norm(A1-A,1)/numpy.linalg.norm(A,1)
        logger.debug('leave ldldown')

    return L, d


#-----
def ldlrk1(L, d, alp, u):
    """
      function L, d, p = ldlrk1(L, d, alp, u)

      Computes LDL^T factorization for LDL^T+alp*uu^T if alp>=0 or if the new factorization is
      definite (both signalled by p=[]); otherwise, the original L,d and a direction p of null
      or negative curvature are returned.

      d contains diag(D) and is assumed positive

      Note: does not work for dimension 0.
    """

    test = 0
    if test:
        logger.debug('enter ldlrk1')
        A = L.dot(diag(d).dot(L.T)) + alp*(u.dot(u.T))

    p = numpy.array([])
    if alp == 0:
        return L, d, p

    n = len(u)
    neps = n*numpy.spacing(1)

    # save old factorization
    L0 = copy.copy(L)
    d0 = copy.copy(d)

    # update
    for k in find(u != 0):
        k = int(k)
        del_ = d[k] + alp*u[k]**2
        if alp < 0 and del_ <= neps:
            # update not definite
            p = numpy.zeros(n)
            p[k] = 1
            p[:k] = numpy.linalg.solve(L[:k,:k].T, p[:k])
            # restore original factorization
            L = L0
            d = d0
            if test:
                indef = ((p.T).dot(A.dot(p))) / ((numpy.abs(p).T).dot((numpy.abs(A)).dot(numpy.abs(p))))
                logger.debug('leave ldlrk1 at 1')

            return L, d, p

        q = d[k]/del_
        d[k] = del_
        # in C, the following 3 lines would be done in a single loop
        # WLAVWLAVWLAV
        ind = numpy.arange(k, n)  # TODO: why n-1 instead of n?
#        if len(L.shape) == 1:
#            L = L.reshape(1, len(L))
        c = L[ind,k].dot(u[k])
        L[ind,k] = numpy.outer(L[ind,k], q).flatten() + numpy.outer((alp*(u[k])/del_), u[ind]).flatten()
        u[ind] = u[ind] - c
        alp = alp*q
        if alp == 0:
            break

    if test:
        A1 = L.dot(diag(d).dot(L.T)), A
        quot = numpy.linalg.norm(A1-A,1)/numpy.linalg.norm(A,1)
        logger.debug('leave ldlrk1 at 2')

    return L, d, p


#-----
def ldlup(L, d, j, g):
    """
      function L, d, p = ldlup(L, d, j, g)

      Updates LDL^T factorization when a unit j-th row and column are replaced by column g
      if the new matrix is definite (signalled by p=[]); otherwise, the original L,d and
      a direction p of null or negative curvature are returned

      d contains diag(D) and is assumed positive

      Note that g must have zeros in other unit rows!!!
    """

    p = numpy.array([])

    test = 0
    if test:
        logger.debug('enter ldlup')
        A = L.dot(diag(d).dot(L.T))
        A[:,j] = g
        A[j,:] = g.T

    n = len(d)
    I = numpy.arange(0, j)
    K = numpy.arange(j+1, n)
    del_ = 0
    if j == 0:
        v = numpy.array([])
        del_ = g[j]
        if del_ <= n*numpy.spacing(1):
            p = numpy.concatenate((1, zeros(n-2)))
            if test:
                logger.debug('A = %s', A)
                logger.debug('p = %s', p)
                Nenner = numpy.abs(p).T.dot(numpy.abs(A).dot(numpy.abs(p)))
                if Nenner == 0:
                    indef1 = 0
                else:
                    indef1 = (p.T.dot(A.dot(p)))/Nenner
                logger.debug('indef1 = %s', indef1)
                logger.debug('leave ldlup at 1')

            return L, d, p

        w = g[K]/del_
        L[j, I] = v.T
        d[j] = del_
        if test:
            A1 = L.dot(diag(d)).dot(L.T)
            logger.debug('A1 = %s', A1)
            logger.debug('A = %s', A)
            logger.debug('quot = %s', numpy.linalg.norm(A1-A, 1)/numpy.linalg.norm(A, 1))
            logger.debug('leave ldlup at 3')

        return L, d, p

    try:
        u = numpy.linalg.solve(L[I[:,None],I], g[I])
    except numpy.linalg.LinAlgError as e:
        # happens if len(I) == 1, so simply divide
        u = g[I]/L[I,I]
    v = u/d[I]
    del_ = g[j] - u.T.dot(v)
    if del_ <= n*numpy.spacing(1):
        try:
            p1 = numpy.linalg.solve(L[I[:,None],I].T, v)
        except numpy.linalg.LinAlgError:
            # happens if len(I) == 1, so simply divide
            p1 = v/L[I,I]
        p = numpy.concatenate((p1, numpy.array((-1,)), numpy.zeros(n-j-1)))
        if test:
            logger.debug('A = %s', A)
            logger.debug('p = %s', p)
            logger.debug('indef1 = %s', (p.T.dot(A.dot(p)))/(numpy.abs(p).T.dot(numpy.abs(A).dot(numpy.abs(p)))))
            logger.debug('leave ldlup at 2')

        return L, d, p

    w = (g[K]-L[K[:,None],I].dot(u))/del_
    L[K[:,None],K], d[K], q = ldlrk1(L[K[:,None],K], d[K], -del_, w.copy())
    if q.size <= 0:
        L[j,I] = v.T
        L[K[:,None],j] = w[:,None]
        d[j] = del_
        if test:
            A1 = L.dot(diag(d).dot(L.T))
            logger.debug('A1 = %s', A1)
            logger.debug('A = %s', A)
            logger.debug('quot = %s', numpy.linalg.norm(A1-A,1)/numpy.linalg.norm(A,1))
            logger.debug('leave ldlup at 4')

    else:
        r"""
        % work around expensive sparse L(K,K)=LKK
        L=[L(1:j,:); LKI,L(K,j),LKK];
        pi=w'*q;
        p=[LII'\(pi*v-LKI'*q);-pi;q];
        """
        # TODO: there is something missing in this mapping (extension of matrices), for now skip
        # this update and let upstream deal with it.
        """
        pi_ = numpy.outer(w.T, q)
        p = numpy.concatenate((numpy.linalg.solve(L[I[:,None],I].T, (pi_*v - L[K[:,None],I].T.dot(q))), - pi_, q))
        if test:
            logger.debug('indef2 = %s', (p.T.dot(A.dot(p)))/(numpy.abs(p).T.dot(numpy.abs(A).dot(numpy.abs(p)))))
            logger.debug('leave ldlup at 5')
        """
        logger.debug('DEBUG: skipping update step in ldlup')

    return L, d, p
