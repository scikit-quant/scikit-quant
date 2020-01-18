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
 function y, f, g, sigma = snoblocf(j, x, f, near, dx, u, v)

 Computes a local fit around the point x0 = x(j,:) and minimizes it
 on a trust region.
 
 Input:
  j             index of the point around which the fit is to be computed
  x             the rows contain the points where the function has been
                evaluated
  f             the corresponding function values and their uncertainties,
                i.e., f[j,0] = f[x[j,:]] and f[j,1] = df[x[j,:]]
  near          near[j,:] is a vector containing the indices of the nearest
                neighbors of the point x[j,:]
  dx            resolution vector, i.e. the ith coordinate of a point to be
                generated is an integer-valued multiple of dx(i)
  u, v          bounds of the box where the points should be generated
 
 Output:
  y             estimated minimizer in the trust region
 
  f1            its estimated function value
  g             estimated gradient for the fit
  sigma         sigma = norm(A*g-b)/sqrt(K-n), where A and b are the
                coefficients resp. right hand side of the fit, n is the
                dimension and K the number of nearest neighbors considered
                (estimated standard deviation of the model errors)
"""

from ._gen_utils import diag, rsort, max_, find, std, maximum_, rand
import numpy


def snoblocf(j, x, f, near, dx, u, v):
    n = u.shape[1]  # dimension of the problem
    x0 = x[j,:]
    f0 = f[j,0]

    df0 = f[j,1]
    D = df0/dx**2
    x1 = x[near[j].astype(int)]
    K = len(x1)
    S = x1-numpy.outer(numpy.ones(K), x0)
    d = 0.5*numpy.abs(S).max(0)
    d = numpy.maximum(d,dx)
    sc = numpy.zeros((1, K))
    for i in range(K):
        sc[0,i] = S[i].dot(diag(D).dot(S[i].T)) + f[int(near[j,i]),1]

    A = S / (sc.T.dot(numpy.ones((1,n))) )
    b = (f[near[j].astype(int),0].reshape(near.shape[1], 1) - f0) / (sc.T)
    U, Sigma, V = numpy.linalg.svd(A, 0)
    V = V.T
    Sigma = diag(Sigma)
    Sigma = numpy.amax(Sigma, axis=1)

    g = V.dot(diag(1./Sigma).dot((U.T).dot(b))).flatten()
    sigma = numpy.sqrt(numpy.sum((A.dot(g)-b.T)**2)/(K-n))

    pl = numpy.maximum(-d, u.flatten()-x0)
    pu = numpy.minimum(d, v.flatten()-x0)
    p = numpy.zeros((n,))
    for i in range(n):
        p[i] = snobqmin(sigma*D[i], g[i], pl[i], pu[i])

    y = snobround(x0+p, u, v, dx)
    nc = 0

    while (numpy.min(numpy.maximum(numpy.abs(x - numpy.outer(numpy.ones(len(x)), y) -
            numpy.outer(numpy.ones(len(x)), dx)), 1)) < 0) and nc < 5:
        p = pl + (pu-pl)*rand(1, n)
        y = snobround(x0+p, u, v, dx)
        nc = nc + 1

    p = y - x0
    err = p.dot(diag(D).dot(p.T)) + df0
    f1 = f0 + p.dot(g) + sigma*err

    return y, f1, g, sigma


#-----
def snobqmin(a, b, xl, xu):
    """
      function x = snobqmin(a, b, xl, xu)

      Minimization of the quadratic polynomial p(x) = a*x^2+b*x over [xl, xu].

      Input:
      a, b      coefficients of the polynomial
      xl, xu    bounds (xl < xu)

      Output:
      x         minimizer in [xl, xu]
    """

    if a > 0:
        x = -0.5*b/a
        x = numpy.minimum(numpy.maximum(xl, x), xu)
    else:
        fl = a*xl**2+b*xl
        fu = a*xu**2+b*xu
        if (fu <= fl).any():
            x = xu
        else:
            x = xl

    return x


#-----
def snobround(x, u, v, dx):
    """
      function x = snobround(x, u, v, dx)

      A point x is projected into the interior of [u, v] and x[i] is
      rounded to the nearest integer multiple of dx[i].

      Input:
      x         vector of length n
      u, v      vectors of length n such that u < v
      dx        vector of length n

      Output:
      x         projected and rounded version of x
    """

    x = numpy.minimum(numpy.maximum(x, u), v)
    dx = dx.reshape(x.shape)
    x = numpy.multiply(numpy.round(numpy.divide(x, dx)), dx)

    i1 = find(x<u)
    for i in i1:
        x[0, i1] += dx[0, i1]

    i2 = find(x>v)
    for i in i2:
        x[0, i2] -= dx[0, i2]

    return x
