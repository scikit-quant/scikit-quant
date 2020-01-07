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
 function local, nlocal = snoblp(f, near, ind)

 Computes a pointer to all "local" points (i.e. points whose neighbors have
 "significantly larger" function values).
 
 Input:
  f            f[j] is the function value of point j
  near         near[j,:] contains the indices of the near.shape[1] neighbors of
 	       point j
  ind          pointer to the boxes to be considered (optional, default 0:len(f)))
 
 Output:
  local        vector containing the indices of all local points
  nlocal       vector containing the indices of all nonlocal points
"""

import numpy


def snoblp(f, near, ind=None):
    if ind is None:
        ind = numpy.arange(len(f))

    local = numpy.array([], dtype=int)
    nlocal = ind.T
    jj = numpy.array([], dtype=int)
    for j in range(len(ind)):
        fmi = numpy.min(f[near[ind[j]]])
        fma = numpy.max(f[near[ind[j]]])
        if f[ind[j]] < fmi - 0.2*(fma-fmi):
            local = numpy.append(local, ind[j])
            jj = numpy.append(jj, j)

    if len(jj):
        nlocal = numpy.delete(nlocal, jj)

    nlocal = nlocal.flatten()
    return local, nlocal
