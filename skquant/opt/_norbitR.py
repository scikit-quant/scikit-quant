from __future__ import division

# Copyright (c) 2014, Stefan Wild All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from SQCommon import Result, ObjectiveFunction
import logging, scipy, scipy.spatial, sys
import numpy as np
from math import pi, cos, sqrt, exp

log = logging.getLogger('SKQ.Orbit')

log.info("""
------------------------------------------------------------------------
S.M. Wild, et al. "Global convergence of radial basis function trust-region
 algorithms for derivative-free optimization. SIAM Review, 55(2):349-371, 2013.
------------------------------------------------------------------------""")

__all__ = ['minimize', 'log']


class OrbitObjectiveFunction(ObjectiveFunction):
    def __init__(self, func, nevals, options = {}):
        options['simple_function'] = True
        ObjectiveFunction.__init__(self, func, options)
        self.nevals = nevals

    def __call__(self, par):
        result = list()
        for i in range(self.nevals):
            result.append(ObjectiveFunction.__call__(self, par))
        return result


def AddPoints_(X,Dist,F,theta2,nmpmax,xkin,delta,maxdelta,rbftype,gamma,ModelIn,Intind):
    """Collects up to nmpmax-n-1 additional points while keeping Hessian bounded.
    Then fits an RBF model to data points (in R^n) using a null space method


    Parameters
    ----------
    X       = [dbl] [nf-by-n] Matrix whose rows are points
    Dist    = [dbl] [nf-by-1] Vector of distances to X(xkin,:)
    F       = [dbl] [nf-by-ns] vector of function values
    theta2  = [dbl] Threshold for adding points
    nmpmax   = [int] Maximum number of model points
    xkin    = [int] Index of current center in X
    delta   = [dbl] Trust region radius
    maxdelta= [dbl] Maximum distance to look for points
    rbftype = [str] RBF type (cubic,multiquadric,Gaussian)
    gamma   = [dbl] Positive RBF parameter
    ModelIn = [int] [nmpmax-by-1] Indices in X of model points
    Intind  = [log] [1-by-nf] Logical indicating whether i is in AffIn

    Returns
    ----------
    nmp      = [int] the number of points currently in ModelIn (<=nmpmax)
    (Not needed) Lambda  = [dbl] [nmp-by-1] vector of coefficients for the RBF part
    (Not needed) Ctail   = [dbl] [n+1-by-1] vector of coefficients for the polynomial tail
    D       = [dbl] [nmpmax-by-n] Matrix of scaled displacement vectors
    """

    nf,n = X.shape # [int,int] Total number of points and dimension
    phi0 = phi_(0,rbftype,gamma,1)[0] # [dbl] Set the constant phi(0) once
    nmp = n + 1 # since we initially have n+1 points in ModelIn.

    # Set up the initial matrices
    L = np.zeros((nmpmax - n - 1,nmpmax - n - 1))
    Z = np.zeros((nmpmax,nmpmax - n - 1))
    D = np.zeros((nmpmax,n))
    PHI = np.zeros((nmpmax,nmpmax))
    Normval = np.zeros(nmpmax+1)
    Q = np.zeros((nmpmax,n+1))
    dummy = np.zeros(nmpmax)
    dummy2 = np.zeros(nmpmax)
    v = np.zeros(nmpmax-n-1)
    ModelIn = np.concatenate((ModelIn,-np.ones(nmpmax-len(ModelIn)))).astype(int)
    D[0,:] = (X[ModelIn[0],:] - X[xkin,:]) / delta

    PHI[0,0] = phi0
    for i in range(1,n + 1):
        D[i,:] = (X[ModelIn[i],:] - X[xkin,:]) / delta
        # Normval[0:i] = scipy.spatial.distance.cdist([D[i,:]],D[0:i,:])
        # PHI[i,0:i] = phi_(Normval[0:i],rbftype,gamma,1)[0]
        # PHI[0:i,i] = PHI[i,0:i].T

    nmp = nf
    return nmp, D 

def AffPoints2_(X,Dist,radius,theta1,xkin):
    """Obtains n linearly indep points of norm<=radius

    Parameters
    ----------
    X      = [dbl] [nf-by-n] Matrix whose rows are points
    Dist   = [dbl] [nf-by-1] Vector of distances to X(xkin,:)
    radius = [dbl] Positive radius
    theta1 = [dbl] Positive validity threshold
    xkin   = [int] Index of current center

    Returns
    -------
    AffIn  = [int] [(n+1)-by-1] Indices in X of linearly independent points
    valid  = [log] Logical set to 1 if valid within radius
    Modeld = [dbl] [1-by-n] Unit model-improving direction
    nmp     = [int] Number of aff indep points (# of nonzeros in AffIn)
    Intind = [log] [1-by-nf] Logical indicating whether i is in AffIn
    """

    nf,n = X.shape
    Modeld = np.zeros(n)               # Initialize for output
    AffIn = -1*np.ones(n + 1)          # vector of integer indices (of size n+1 for output)
    Intind = np.zeros(nf,dtype=bool)   # vector of indicators for indices in AffIn
    nmp = 0                            # number of initial l.i. points

    Q = np.eye(n) # Get initial null space
    R = np.zeros((n,n))
    for ind in np.arange(nf-1,-1,-1):
        if Dist[ind] <= radius: # Only look at the nearby points
            D = (X[ind,:] - X[xkin,:]) / radius # Current displacement
            proj = np.linalg.norm(D.dot(Q[:,nmp:n])) # [double] # Note that Q(:,nmp+1:n) is a matrix
            if (proj >= theta1): # add this index to AffIn
                nmp = nmp + 1
                AffIn[nmp-1] = ind
                Intind[ind] = True
                if (nmp == n):
                    valid = True
                    return (AffIn,valid,Modeld,nmp,Intind)

                # Update QR factorization:
                R[:,nmp-1] = D.dot(Q) # add D
                for k in np.arange(n-1,nmp-1,-1):
                    G,R[[k-1,k],nmp-1] = planerot(R[[k-1,k],nmp-1])
                    Q[:,[k-1,k]] = Q[:,[k-1,k]].dot(G.T)
    #if you get out of this loop then nmp<n
    Modeld = Q[:,nmp:n].T
    valid = False
    return (AffIn,valid,Modeld,nmp,Intind)

def boxline_(D,X,L,U):
    """ This routine finds the smallest t>=0 for which X+t*D hits the box [L,U]
    Parameters
    ----------
    D      = [dbl] [n-by-1] Direction
    L      = [dbl] [n-by-1] Lower bounds
    X      = [dbl] [n-by-1] Current Point (assumed to live in [L,U])
    U      = [dbl] [n-by-1] Upper bounds

    Returns
    ----------
    t      = [dbl] Value of smallest t>=0 for which X+t*D hits a constraint. Set to 1 if t=1 does not hit constraint for t<1.
    """

    n = len(X)
    t = 1
    for i in range(0,n):
        if D[i] > 0:
            t = min(t,(U[i] - X[i]) / D[i])
        else:
            if D[i] < 0:
                t = min(t,(L[i] - X[i]) / D[i])
    return t

def boxproj_(z,p,l,u):
    """ This subroutine projects the vector z onto the box [l,u]

    z,l,u are vectors of length p
    """

    z = z.flatten()
    for i in range(0,p):
        z[i] = min(max(l[i],z[i]),u[i])
    return z

def CheckPoised_(X,Dist,theta1,maxdelta,xkin,radius,nmp,Intind,ModelIn):
    """Obtains additional affine indep points and generates geom-imp pts if nec

    Parameters
    ----------
    X      = [dbl] [nf-by-n] Matrix whose rows are points
    Dist   = [dbl] [nf-by-1] Vector of distances to X(xkin,:)
    theta1 = [dbl] Positive validity threshold
    maxdelta = [dbl] Maximum distance to look for points
    xkin   = [int] Index of current center
    radius = [dbl] Positive radius
    ModelIn = [int] [(n+1)-by-1] Indices in X of linearly independent points
    nmp      = [int] Number of Model points (# of nonzeros in ModelIn)
                        Note: nmp<n before we call this
    Intind  = [log] [1-by-nf] Logical indicating whether i is in AffIn


    Returns
    ----------
    ModelIn = [int] [(n+1)-by-1] Indices in X of linearly independent points
    nmp      = [int] Number of Model points (# of nonzeros in ModelIn)
                        Note: nmp<n before we call this
    Intind  = [log] [1-by-nf] Logical indicating whether i is in AffIn
    GPoints = [dbl] [n-by-n] Matrix of (n-nmp) points to be evaluated
    """

    ModelIn = np.array(ModelIn,dtype='int')

    nf,n = X.shape
    GPoints = np.zeros((n,n))

    if nmp is not 0:
        R = (X[ModelIn[0:nmp],:] - np.repeat([X[xkin,:]],nmp,axis=0)) / radius # The points we have so far
        Q,R = np.linalg.qr(R.T,mode='complete') # Get QR of points so far
        R = np.atleast_2d(R)
        R = np.hstack((R,np.zeros((n,n-R.shape[1]))))
    else:
        Q = np.eye(n)
        R = np.zeros((n,n))

    for ind in np.arange(nf-1,-1,-1):
        if (Intind[ind] == False) and (Dist[ind] <= maxdelta):
            D = (X[ind,:] - X[xkin,:]) / radius
            proj = np.linalg.norm(D.dot(Q[:,nmp:n])) # [double] # Note that Q(:,nmp+1:n) is a matrix
            if (proj >= theta1):
                nmp = nmp + 1
                ModelIn[nmp-1] = ind
                Intind[ind] = True
                if nmp == n:
                    return (ModelIn,nmp,Intind,GPoints)
                R[:,nmp-1] = (D.dot(Q)).T
                for j in np.arange(n-1,nmp-1,-1):
                    G = planerot(R[[j-1,j],nmp-1])[0]
                    R[[j-1, j],nmp-1] = G.dot(R[[j-1,j],nmp-1])
                    Q[0:n,[j-1,j]] = Q[0:n,[j-1,j]].dot(G.T)
    # if you get out of this loop then nmp<n
    GPoints[0:n - nmp,:] = Q[:,nmp:n].T
    return ModelIn, nmp, Intind, GPoints

def ORBIT2(func,rbftype,gamma,n,nfmax,nmpmax,delta,maxdelta,trnorm,gtol,Low,Upp,nfs,X,F,xkin,ns,
           start = 6, end = 20, methodLa = 'nn', sep = True):
    """ Optimization by Radial Basis function Interpolation in Trust Regions
    This is a translation of the MATLAB version of ORBIT found `here <http://www.mcs.anl.gov/~wild/orbit/>`_ and assumes that the initial point is X(xkin,:) and that it's been evaluated

    Parameters
    ----------
    Low  = [dbl] [1-by-n] vector of lower bounds
    Upp  = [dbl] [1-by-n] vector of lower bounds
    nfs  = [int] size of initial X and F
    X    = [dbl] [nfs+nfmax-by-n] matrix of locations
    F    = [dbl] [nfs+nfmax-by-ns] vector of function values
    xkin = [int] Index of initial point
    ns = [int] Number of runs for each new design
    start, end, methodLa, sep = see GPsurface
    """

    if trnorm == 0:
        trnorm = np.inf
    exitflag = 0

    # Set trust-region RBF algorithm parameters and initialize output
    mindelta = min(1e-07,(1e-05) * delta) # Minimum tr radius (technically must be 0)
    eta0 = 0 # Parameters for decrease (0<eta0<eta1<1)
    eta1 = 0.2
    gam0 = 2 # Parameters for changing delta (gam0,gam1> 1)
    gam1 = 2
    c1 = 10 # Factor for checking validity
    c2 = sqrt(n) * maxdelta # Maximum distance for adding points
    theta1 = 0.001 # Pivot threshold for validity
    theta2 = 1e-07 # Pivot threshold for additional points
    alpha = 0.9 # Shrinkage parameter
    X = np.vstack((X,np.zeros((nfmax+10,n)))) # Stores the evaluation point locations
    F = np.vstack((F,np.zeros((nfmax+10,ns)))) # Stores the function values of evaluated points
    Dist = np.zeros((nfs + nfmax+10)) # Stores displacement distances

    Intind = np.zeros(nfs + nfmax,dtype=bool) # Stores indicators for model points
    xkin_mat = np.nan*np.ones((nfmax,n)) # Stores the xkin value at each iteration
    xkin_val = np.nan*np.ones(nfmax) # Stores xkin's function value

    nf = 0

    # Make sure that we are sufficiently in the interior of the domain:
    m1 = np.min(np.max(np.vstack((X[xkin,:] - Low,Upp - X[xkin,:])),axis=0)) # make sure you understand what this is doing
    if (m1 < mindelta): # If infeasible or bounds too tight relative to mindelta
        log.info('Minimum trust-region radius too large given the bounds')
        return (X,F,xkin,nf,exitflag,xkin_mat,xkin_val)
    else:
        if (m1 < 0.5 * delta):
            delta = m1
            log.info('Changing initial radius to %g delta ='%delta)

    while (nf < nfmax):
        # STEP 1: Find affinely independent points & check if fully linear
        AffIn,valid,Modeld,nmp,Intind[0:nfs + nf] = AffPoints2_(X[0:nfs + nf,:],Dist,c1 * delta,theta1,xkin)
        if (not valid): # Model is not valid, check if poised
            AffIn,nmp,Intind[0:nfs + nf],GPoints = CheckPoised_(X[0:nfs + nf,:],Dist,theta1,c2,xkin,c1 * delta,nmp,Intind[0:nfs + nf],AffIn)
            if nmp < n: # Need to include additional points to obtain a model
                T1 = np.zeros(n-nmp)
                T2 = np.zeros(n-nmp)
                for j in range(0,n - nmp):
                    GPoints[j,:] = GPoints[j,:] / np.linalg.norm(GPoints[j,:],ord=trnorm) # Make unit length.
                    T1[j] = boxline_(GPoints[j,:],X[xkin,:],Low,Upp)
                    T2[j] = boxline_(- GPoints[j,:],X[xkin,:],Low,Upp)
                if min(np.max(np.vstack((T1,T2)),axis=0)) < theta1 * delta * c1:
                    for j in range(0,min(n,nfmax - nf)):
                        t1 = Low[j] - X[xkin,j]
                        t2 = Upp[j] - X[xkin,j]
                        if t2 > - t1:
                            t1 = min(t2,delta)
                        else:
                            t1 = max(t1,- delta)
                        nf = nf + 1
                        X[nfs + nf,:] = X[xkin,:]
                        X[nfs + nf,j] = max(Low[j],min(X[xkin,j] + t1,Upp[j])) # added min and max to make sure in bounds
                        F[nfs + nf,:] = func(X[nfs + nf,:])
                        Dist[nfs + nf] = abs(t1)
                else: # Safe to use our directions:
                    for j in range(0,min(n - nmp,nfmax - nf)):
                        if T1[j] >= theta1 * delta * c1:
                            X[nfs + nf,:] = boxproj_(X[xkin,:] + min(T1[j],delta) * GPoints[j,:],n,Low,Upp) # added projection to make sure in bounds
                        elif T2[j] >= theta1 * delta * c1:
                            X[nfs + nf,:] = boxproj_(X[xkin,:] - min(T2[j],delta) * GPoints[j,:],n,Low,Upp) # added projection to make sure in bounds
                        F[nfs + nf,:] = func(X[nfs + nf,:])
                        Dist[nfs + nf] = np.linalg.norm(X[nfs + nf,:] - X[xkin,:],ord=trnorm)
                        nf = nf + 1
            if nf >= nfmax: #Budget has been exhausted
                X = X[0:nf + nfs]; F = F[0:nf + nfs]; xkin_mat = xkin_mat[0:nf,:]; xkin_val = xkin_val[0:nf];
                return (X,F,xkin,nf,exitflag,xkin_mat,xkin_val)
            AffIn,nmp,Intind[0:nfs + nf],GPoints = CheckPoised_(X[0:nfs + nf,:],Dist,theta1,c2,xkin,c1 * delta,nmp,Intind[0:nfs + nf],AffIn)
        nmp = nmp + 1

        #Add xkin to FRONT of AffIn
        AffIn[1:nmp] = AffIn[0:nmp-1]
        AffIn[0] = xkin
        Intind[xkin] = True

        # STEP 2&3: Collect additional points and obtain Model Parameters
        out_tup = AddPoints_(X[0:nfs + nf,:],Dist[0:nfs + nf],F[0:nfs + nf,:],theta2,nmpmax,xkin,delta,c2,rbftype,gamma,AffIn,Intind[0:nfs + nf])
        nmp = out_tup[0]; #Lambda = out_tup[1]; Ctail = out_tup[2];
        D = out_tup[1]
        # rbfc,Rbfgrad = RBFsurface_(np.zeros(n),rbftype,gamma,Lambda[0:nmp],Ctail,D[0:nmp,:],2)[0:2] # model at center


        rbfc,Rbfgrad = GPsurface_(X = X[xkin,:], Xtrain=X[0:nfs + nf,:], Ftrain=F[0:nfs + nf,:], start=start, end=end,
                                  lower=0.001 * np.ones(n), upper=np.ones(n), covtype="Gaussian", numout=2,
                                  method=methodLa, sep = sep)[0:2] # model at center
        Rbfgrad = Rbfgrad * delta # Take into account rescaling


        # STEP 4: Final Criticality Test
        ng = 0
        for i in range(0,n):
            if Rbfgrad[i] > 0 and (X[xkin,i] > Low[i]):
                ng = ng + Rbfgrad[i] ** 2
            else:
                if Rbfgrad[i] < 0 and (X[xkin,i] < Upp[i]):
                    ng = ng + Rbfgrad[i] ** 2
        ng = sqrt(ng)
        while (ng <= gtol) and (valid):

            log.info('******** RBF gradient is smaller than tolerance! *********')
            X = X[0:nf + nfs]; F = F[0:nf + nfs,:]; xkin_mat = xkin_mat[0:nf,:]; xkin_val = xkin_val[0:nf];
            exitflag = 1;
            return (X,F,xkin,nf,exitflag,xkin_mat,xkin_val)

        # STEP 5: Compute a Step
        Xsp,mdec = SP4_(D[0:nmp,:],trnorm,X[xkin,:],Low,Upp,delta,
                        xkin=xkin, F=F, nfs=nfs, nf=nf, Xtrain = X, start = start, end = end, methodLa =methodLa, sep = sep) # added by Mickael
        X[nfs + nf,:] = boxproj_(X[xkin,:] + delta * Xsp,n,Low,Upp) # Added projection to ensure feasibility
        F[nfs + nf,:] = func(X[nfs + nf,:])
        Dist[nfs + nf] = delta * np.linalg.norm(Xsp,ord=trnorm)

        # STEP 6: Update trust region parameters

        # rho v1: use observations only
        # rho = (combine_fvals(F[xkin]) - combine_fvals(F[nfs + nf])) / mdec # ratio of the actual decrease to the decrease predicted by the model

        # rho v2: use predictions only
        pred_xin, __, __, __, __, sd2in = GPsurface_(X=X[xkin,:], Xtrain=X[0:(nfs + nf + 1),:], Ftrain=F[0:(nfs + nf + 1),:], start=start, end=end,
                              lower = 0.001 * np.ones(n), upper=np.ones(n), numout=1, method=methodLa, sep=sep, sd2=True)

        pred_xne, __, __, __, __, sd2ne = GPsurface_(X=X[nfs + nf,:], Xtrain=X[0:(nfs + nf + 1),:], Ftrain=F[0:(nfs + nf + 1),:], start=start, end=end,
                              lower = 0.001 * np.ones(n), upper=np.ones(n), numout=1, method=methodLa, sep=sep, sd2=True)


        # rho = (pred_xin - pred_xne) / mdec

        # rho v3: use both predictions and observations (see Kaminski2015 and reference therein)
        o_xin = np.mean(F[xkin]); os2_in = max(1.5e-16, np.var(F[xkin]))  # empirical mean and variance at x_in
        o_xne = np.mean(F[nfs + nf]); os2_ne = max(1.5e-16, np.var(F[nfs + nf]))  # empirical mean and variance at x_new
        sd2in = max(1.5e-16, sd2in); sd2ne = max(1.5e-16, sd2ne)  # avoid zero variance

        f_in = (pred_xin/sd2in + ns*o_xin/os2_in)/(1/sd2in + ns/os2_in)  # fused mean at x_in
        f_ne = (pred_xne/sd2ne + ns*o_xne/os2_ne)/(1/sd2ne + ns/os2_ne)  # fused mean at x_ne
        rho = (f_in - f_ne) / mdec

        if (rho >= eta1) or ((rho > eta0) and (valid)): # Accept iterate
            xkin = nfs + nf # Update center of trust region
            xkin_val[nf] = pred_xne
            if trnorm == np.inf:
                Dist[0:nfs+nf+1] = scipy.spatial.distance.cdist([X[xkin,:]], X[0:nfs+nf+1,:],'chebyshev')
            else:
                sys.exit('trnorm must currently be 0')
        else:
            xkin_val[nf] = pred_xin

        xkin_mat[nf,:] = X[xkin,:]
        # xkin_val[nf] = combine_fvals
        # xkin_val[nf] = GPsurface_(X=X[xkin,:], Xtrain=X[0:nfs + nf,:], Ftrain=F[0:nfs + nf,:], start=start, end=end,
        #                           lower=0.001 * np.ones(n), upper=np.ones(n), numout=1, method=methodLa, sep=sep)[0]
        nf = nf + 1
        if (rho >= eta1) and (np.linalg.norm(Xsp,ord=trnorm) > 0.5): # Expand trust region
            delta = min(delta * gam1,maxdelta)
        elif (rho >= eta1): # Maintain trust region
            pass
        elif (valid): # Shrink trust region
            delta = max(delta / gam0,mindelta)
        else: # STEP 7: Maintain trust region and Improve model
            if nf < nfmax: # If it fits in the budget, help model be more valid
                A = np.zeros(Modeld.shape[0])
                for md in range(0,Modeld.shape[0]):
                    Modeld[md,:] = Modeld[md,:] / np.linalg.norm(Modeld[md,:],ord=trnorm) # Make unit length.
                    # A[md] = RBFsurface_(Modeld[md,:],rbftype,gamma,Lambda[0:nmp],Ctail,D[0:nmp,:],1)[0]
                    # m2 = RBFsurface_(- Modeld[md,:],rbftype,gamma,Lambda[0:nmp],Ctail,D[0:nmp,:],1)[0]
                    A[md] = GPsurface_(X = X[xkin,:] + delta * Modeld[md,:], Xtrain=X[0:nfs + nf,:], Ftrain=F[0:nfs + nf,:],
                                       lower=0.001 * np.ones(n), upper=np.ones(n), covtype="Gaussian", numout=1,
                                       start=start, end=end, method=methodLa, sep=sep)[0]
                    m2 = GPsurface_(X = X[xkin,:] - delta * Modeld[md,:], Xtrain=X[0:nfs + nf,:], Ftrain=F[0:nfs + nf,:],
                                    lower=0.001 * np.ones(n), upper=np.ones(n), covtype="Gaussian", numout=1,
                                    start=start, end=end, method=methodLa, sep=sep)[0]
                    if m2 < m1:
                        Modeld[md,:] = - Modeld[md,:]
                        A[md] = m2
                # Evaluate model directions:
                indo = np.argsort(A)
                for i in range(0,Modeld.shape[0]):
                    md = indo[i]
                    t1 = boxline_(Modeld[md,:],X[xkin,:],Low,Upp)
                    t2 = boxline_(- Modeld[md,:],X[xkin,:],Low,Upp)
                    if t1 >= theta1 * delta:
                        X[nfs + nf,:] = X[xkin,:] + min(t1,delta) * Modeld[md,:]
                        break
                    elif t2 >= theta1 * delta:
                        X[nfs + nf,:] = X[xkin,:] - min(t2,delta) * Modeld[md,:]
                        break
                    elif md == indo[-1]:
                        t1 = 0
                        tvec = np.zeros(min(n,nfmax-nf))
                        for j in range(0,min(n,nfmax - nf)):
                            tvec[j] = max(Low[j] - X[xkin,j],- delta)
                            t2 = Upp[j] - X[xkin,j]
                            if t2 > - tvec[j]:
                                tvec[j] = min(t2,delta)
                            if t1 <= abs(Modeld[md,j] * tvec[j]):
                                t1 = abs(Modeld[md,j] * tvec[j])
                                jstar = j
                        X[nfs + nf,:] = X[xkin,:]
                        X[nfs + nf,jstar] = max(Low[jstar],min(X[xkin,jstar] + tvec[jstar],Upp[jstar])) # added projection to esnure feasible
                F[nfs + nf] = func(X[nfs + nf,:])
                Dist[nfs + nf] = np.linalg.norm(X[nfs + nf,:] - X[xkin,:],ord=trnorm)
                nf = nf + 1

    X = X[0:nf + nfs]; F = F[0:nf + nfs]; xkin_mat = xkin_mat[0:nf,:]; xkin_val = xkin_val[0:nf];
    return (X,F,xkin,nf,exitflag,xkin_mat,xkin_val)

def phi_(r,rbftype,gamma,numout):
    """Evaluates the RBF functions phi (phi', phi'') at a vector of values

    Parameters
    ----------
    r      = [p-by-1] vector of nonnegative floats
    rbftype= [string] RBF type (cubic,multiquadric,Gaussian)
    gamma  = [scalar] (0 = default) positive RBF parameter
    numout = [integer] indicates how many outputs we want [1,2,3]

    Returns
    ----------
    Out1   = [p-by-1] vector of values phi(r)
    Out2   = [p-by-1] optional vector of values phiprime(r)
    Out3   = [p-by-1] optional vector of values phi2prime(r)
    """

    p = len(np.atleast_1d(r))
    Out1 = np.zeros(p)
    Out2 = np.zeros(p)
    Out3 = np.zeros(p)

    if 'cubic' == rbftype:
        Out1 = r ** 3
    else:
        if 'multiquadric' == rbftype:
            Out1 = - sqrt(r ** 2 + (gamma * np.ones(len(r))) ** 2)
        else:
            if 'Gaussian' == rbftype:
                Out1 = exp(- (r / (gamma * np.ones(len(r)))) ** 2)
            else:
                log.error('Error: Unknown type.')
                return (Out1,Out2,Out3)
    if numout > 1:
        if 'cubic' == rbftype:
            Out2 = 3 * (r ** 2)
        else:
            if 'multiquadric' == rbftype:
                Out2 = - r / sqrt(r ** 2 + (gamma * np.ones(len(r))) ** 2)
            else:
                if 'Gaussian' == rbftype:
                    Out2 = - ((2 * r) / ((gamma * np.ones(len(r))) ** 2)).dot(exp(- (r / (gamma * np.ones(len(r)))) ** 2))
    if numout > 2:
        if 'cubic' == rbftype:
            Out3 = 6 * r
        else:
            if 'multiquadric' == rbftype:
                Out3 = - 1.0 / sqrt(r ** 2 + (gamma * np.ones(len(r))) ** 2) + (r ** 2) / ((r ** 2 + (gamma * np.ones(len(r))) ** 2) ** 1.5)
            else:
                if 'Gaussian' == rbftype:
                    Out3 = (4 * r ** 2 * (gamma ** - 4) - 2.0 / ((gamma * np.ones(len(r))) ** 2)).dot(exp(- (r / (gamma * np.ones(len(r)))) ** 2))

    return (Out1,Out2,Out3)

def phiprimezero_(rbftype,gamma):

    if 'cubic' == rbftype:
        Out = 0
    else:
        if 'multiquadric' == rbftype:
            Out = 1 / gamma
        else:
            if 'Gaussian' == rbftype:
                Out = - 2 / (gamma ** 2)
            else:
                log.error('Error: Unknown type.')
                return Out
    return Out

def projgrad_(H,C,L,U,n,theta):
    """Computes an approximate solution to the tr sp with sufficient decrease.
    Follows Algorithm 16.5: Gradient Projection Method for QPs (N&W)

    Parameters
    ----------
    H      = [dbl] [n-by-n] Symmetric Hessian Matrix
    C      = [dbl] [n-by-1] Gradient Vector
    L      = [dbl] [n-by-1] Vector of lower bounds
    U      = [dbl] [n-by-1] Vector of upper bounds
    n      = [int] Dimension (number of continuous variables)
    theta  = [dbl] Stopping criterion for resid (should be <= omega_star)

    Returns
    ----------
    X      = [dbl] [n-by-1] Approximate SP solution
    """

    X = 0.5 * (L + U) # Initial point
    for numit in range(0,100):
        G = H.dot(X) + C # Current gradient

        # Step 0: Find the Breakpoints and update the residual
        T = np.zeros(n) # Initialize breakpoint vector
        resid = 0 # Initialize residual
        for i in range(0,n):
            if G[i] < 0 and X[i] < U[i]:
                T[i] = (X[i] - U[i]) / G[i]
                resid = resid + G[i] ** 2
            elif G[i] > 0 and X[i] > L[i]:
                T[i] = (X[i] - L[i]) / G[i]
                resid = resid + G[i] ** 2
            elif G[i] == 0:
                T[i] = np.inf
        if sqrt(resid) < theta: # Solution found
            return X
        B = np.argsort(T); T = T[B] # Sort the T values and corresponding indices

        # Step 1: Find the Cauchy Point
        tt = 0 # Time counter
        G = - G # Look at the steepest descent direction
        A = np.zeros(n,dtype=bool) # Indicator of the active constraints
        for i in range(0,n):
            if (T[i] == np.inf):
                break
            if (T[i] > tt):
                f1 = G.T.dot(C + H.dot(X))
                f2 = G.T.dot(H.dot(G))
                if f1 > 0:
                    break # There is a local minimizer at X
                elif (f2 != 0) and (-f1/f2 < T[i]-tt) and (-f1/f2 > 0):
                    X = X - (f1 / f2) * G # local minimizer
                    break
                # Update:
                X = X + (T[i] - tt) * G
                tt = T[i]
            G[B[i]] = 0 # Zero the active constraint
            A[B[i]] = True # Indicate that constraint i is active
        m = np.sum(~A) # Number of inactive constraints

        # Step 2: DO CG
        if (m > 0): # Only do if there are inactive constraints remaining
            Xz = X[~A]
            Cz = C[~A] + H[np.ix_(~A,A)].dot(X[A])
            Hz = H[np.ix_(~A,~A)]
            Lz = L[~A]
            Uz = U[~A]
            R = Hz.dot(Xz) + Cz
            Z = -np.copy(R)

            k = 0 # Initialize iteration counter
            while np.linalg.norm(R) >= 1e-10 and k < 5:
                k = k + 1 # Increment iteration counter
                zhz = Z.dot(Hz.dot(Z)) # Compute quadratic form
                if (zhz <= 0):
                    for i in range(0,m):
                        if Z[i] > 0:
                            T[i] = (Uz[i] - Xz[i]) / Z[i]
                        else:
                            if Z[i] < 0:
                                T[i] = (Lz[i] - Xz[i]) / Z[i]
                            else:
                                T[i] = np.inf
                    Xz = Xz + min(T[0:m]) * Z # Important to only look at T(1:m)
                    break
                Z = (R.T.dot(R)/zhz)*(Z) # Scale direction here to save flops
                if np.linalg.norm(Z - boxproj_(Z,m,Lz - Xz,Uz - Xz)) > 0:
                    for i in range(0,m):
                        if Z[i] > 0:
                            T[i] = (Uz[i] - Xz[i]) / Z[i]
                        else:
                            if Z[i] < 0:
                                T[i] = (Lz[i] - Xz[i]) / Z[i]
                            else:
                                T[i] = np.inf
                    Xz = Xz + min(T[0:m]) * Z # Important to only look at T(1:m)
                    break
                Xz = Xz + Z
                Rnew = R + Hz.dot(Z)
                Z = - Rnew + (zhz * (Rnew.dot(Rnew))/(R.dot(R))**2)*(Z)
                R = np.copy(Rnew)
            if k > 0: # k=0 when R was small enough and we didn't do anything
                X[~A] = Xz # Set inactive components to Xz
    return X

def SP4_(D,trnorm,X,Low,Upp,delta,
         start, end, methodLa, sep, xkin=None, F=None, nfs=None, nf=None, Xtrain=None):  # to be passed to GPsurface_

    n = D.shape[1]     # [dbl] Problem dimension
    # armijo = 0.0001  # [dbl] Armijo parameter (take to be small [<.5])
    # eta_star = 0.001   # [dbl] Convergence tolerance
    kmax = 20          # [int] Maximum number of iterations
    omega_star = 1e-07 # [dbl] Convergence tolerance
    radmax = 0.5       # [dbl] Bound on the trust region radius

    dinit = None  # to reuse hyperparameters
    ginit = None  # to reuse hyperparameters

    # Compute value of model and its gradient at Xk:
    # rbfc,Rbfgrad = RBFsurface_(np.zeros((1,n)),rbftype,gamma,Lambda,Ctail,D,2)[0:2]

    rbfc, Rbfgrad, __, dinit, ginit, __ = GPsurface_(X=Xtrain[xkin, :], Xtrain=Xtrain[0:nfs + nf, :], Ftrain=F[0:nfs + nf, :],
                                                 lower=0.001 * np.ones(n), upper=np.ones(n), covtype="Gaussian", numout=2,
                                                 start=start, end=end, method=methodLa, sep=sep, ginit=ginit, dinit=dinit)
    Rbfgrad = Rbfgrad * delta # Take into account scaling

    # Get Cauchy Point (backtracking line search)------------------------------
    t = 1 / np.linalg.norm(Rbfgrad)
    tmin = 0
    tmax = np.inf
    for j in range(0,100):
        P = boxproj_(X - t * delta * Rbfgrad.T,n,Low,Upp)
        S = (P - X) / delta # Proposed Step
        nS = np.linalg.norm(S,ord=trnorm) # Norm of proposed step
        innerprod = S.dot(Rbfgrad) # A useful inner product
        projtan =  np.linalg.norm(Rbfgrad * [(P.T - boxproj_(P.T - Rbfgrad,n,Low,Upp)) != 0]) # Projection onto tangent cone
        # rbfval = RBFsurface_(S,rbftype,gamma,Lambda,Ctail,D,1)[0] # Evaluate RBF model
        rbfval, __, __, dinit, ginit = GPsurface_(X=P, Xtrain=Xtrain[0:nfs + nf, :], Ftrain=F[0:nfs + nf, :],
                            lower=0.001 * np.ones(n), upper=np.ones(n), covtype="Gaussian", numout=1,
                            start=start, end=end, method=methodLa, sep=sep, dinit=dinit, ginit=ginit)[0:5]

        if (nS > 1) or (rbfval > rbfc + 0.75 * innerprod):
            tmax = t
            t = 0.5 * (tmax + tmin)
        elif (nS < 0.5) and (rbfval < rbfc + 0.25 * innerprod) and (projtan > 0.1 * abs(innerprod)):
            tmin = t
            if tmax == np.inf:
                t = 2 * t
            else:
                t = 0.5 * (tmax + tmin)
        else:
            break # exit the for loop with the cauchy point in hand
    SCP = np.copy(S) # Save the Cauchy Point
    rbftau = rbfval

    ### OLD:
    if (trnorm == np.inf): # The infinity norm case
        Lower = np.concatenate((np.maximum((Low - X)/delta,-1),[0]))
        Upper = np.concatenate((np.minimum((Upp - X)/delta, 1),[1]))
        # rbfval,Grad,Hess = RBFsurface_(S[0:n].T,rbftype,gamma,Lambda,Ctail,D,3)[0:3]

        rbfval, Grad, Hess, dinit, ginit = GPsurface_(X=Xtrain[xkin,:] + delta*S, Xtrain=Xtrain[0:nfs + nf, :], Ftrain=F[0:nfs + nf, :],
                                        lower=0.001 * np.ones(n), upper=np.ones(n), covtype="Gaussian", numout=3,
                                        start=start, end=end, method=methodLa, sep=sep, dinit=dinit, ginit=ginit)[0:5]
        # Take into account scaling
        Grad = Grad * delta
        Hess = Hess * delta

        psi = np.linalg.norm(boxproj_(- Grad,n,Lower[0:n] - S[0:n],Upper[0:n] - S[0:n]))
        rad = 0.5 # Initial trust-region radius
        trit = 0 # Initialize the number of trust-region iterations
        while (psi > omega_star) and (trit <= kmax): # Find an approximate subproblem solution:
            trit = trit + 1
            Trlower = np.maximum(Lower[0:n] - S[0:n],- rad)
            Trupper = np.minimum(Upper[0:n] - S[0:n],rad)

            # Compute a step via the Gradient Projection Method with TRs
            Sd = projgrad_(Hess,Grad,Trlower,Trupper,n,0.5 * omega_star)
            # actred = (RBFsurface_((S[0:n] + Sd[0:n]).T,rbftype,gamma,Lambda,Ctail,D,1)[0] - rbfval)

            tmp, __, __, dinit, ginit = GPsurface_(X=Xtrain[xkin,:] + delta*((S[0:n] + Sd[0:n]).T), Xtrain=Xtrain[0:nfs + nf, :],
                                 Ftrain=F[0:nfs + nf, :], lower=0.001 * np.ones(n), upper=np.ones(n),
                                 covtype="Gaussian", numout=1, method=methodLa, sep=sep, start=start, end=end,
                                 dinit=dinit, ginit=ginit)[0:5]
            actred = tmp - rbfval

            rho = actred / (0.5 * Sd.dot(Hess.dot(Sd)) + (Grad.T + S[0:n].dot(Hess)).dot(Sd))
            if rho == 0:
                break

            # Update trust-region radius
            if rho < 0.25:
                rad = 0.25 * rad
            elif rho > 0.75 and np.linalg.norm(Sd) > 0.99 * rad:
                rad = min(2 * rad,radmax)

            # Update current point
            if rho > 0.01:
                S = S + Sd
                # rbfval,Grad,Hess = RBFsurface_(S[0:n].T,rbftype,gamma,Lambda,Ctail,D,3)[0:3]
                rbfval, Grad, Hess, dinit, ginit = GPsurface_(X=Xtrain[xkin, :] + delta * S, Xtrain=Xtrain[0:nfs + nf, :],
                                                Ftrain=F[0:nfs + nf, :], lower=0.001 * np.ones(n), upper=np.ones(n),
                                                covtype="Gaussian", numout=3, start=start, end=end, method=methodLa,
                                                              sep=sep, dinit=dinit, ginit=ginit)[0:5]
                # Account for rescaling
                Grad = Grad * delta
                Hess = Hess * delta

                psi = np.linalg.norm(boxproj_(- Grad,n,Lower[0:n] - S[0:n],Upper[0:n] - S[0:n]))

    else:
        sys.exit("The current version is only for bound-constrained problems, which requires trnorm=0")

    S = S[0:n] # return S
    if rbfval > rbftau: # Make sure decrease is better than CP
        S = SCP.T
    # mdec = rbfc - RBFsurface_(S,rbftype,gamma,Lambda,Ctail,D,1)[0]

    mdec = rbfc - GPsurface_(X=Xtrain[xkin, :] + delta * S, Xtrain=Xtrain[0:nfs + nf, :], Ftrain=F[0:nfs + nf, :],
                             lower=0.001 * np.ones(n), upper=np.ones(n), covtype="Gaussian", numout=1,
                             start=start, end=end, method=methodLa, sep=sep, dinit=dinit, ginit=ginit)[0]
    return (S,mdec)

def planerot(x):
    if x[1] != 0:
        r = np.linalg.norm(x)
        G = np.vstack((x,np.array([-x[1], x[0]])))/r
        x = np.array([r, 0])
    else:
        G = np.eye(2)
    return (G, x)

# Connexion with R GP packages
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

laGP = importr('laGP')

# Define gradient and Hessian of laGP prediction
robjects.r('''
#' Gradient of the predictive mean
#' @param model output from laGP
#' @param x design location for prediction
#' @param X locations considered in the model
#' @param Z response at X
dmean <- function(model, x, X, Z){
  if(length(model$mle) == 4){
    d <- as.numeric(model$mle[1])
  }else{
    d <- as.numeric(model$mle[1:ncol(X)])
  }
  g <- model$mle$g
  dk <- -2 * (matrix(x / d, nrow(X), ncol(X), byrow = T) - X %*% diag(1/d, ncol(X))) * matrix(hetGP::cov_gen(X1 = x, X2 = X, theta = d), nrow(X), ncol(X)) 
  K <- hetGP::cov_gen(X1 = X, theta = d) + diag(g, nrow(X))
  return(drop(crossprod(dk, chol2inv(chol(K))) %*% Z))
}

#' Hessian of the predictive mean
#' @param grad to return list with both gradient and Hessian
Hmean <- function(model, x, X, Z, grad = TRUE){
  if(length(model$mle) == 4){
    d <- as.numeric(model$mle[1])
  }else{
    d <- as.numeric(model$mle[1:ncol(X)])
  }
  g <- model$mle$g
  
  h <- (matrix(x/d, nrow(X), ncol(X), byrow = T) - X %*% diag(1/d, ncol(X)))
  
  KiZ <- chol2inv(chol(hetGP::cov_gen(X1 = X, theta = d) + diag(g, nrow(X)))) %*% Z
  k <- hetGP::cov_gen(X1 = x, X2 = X, theta = d)
  
  # diag part (less efficient if separable)
  d2k <- (4*h^2 - 2 * matrix(1/d, nrow(X), ncol(X), byrow = T)) * matrix(k, nrow(X), ncol(X))
  res <- diag(drop(crossprod(d2k, KiZ)))
  
  if(length(d) == 1) d <- rep(d, ncol(X))
  
  for(i in 1:(ncol(X) - 1)){
    for(j in (i+1):ncol(X)){
      res[i,j] <- res[j,i] <- drop((4 * h[, i] * h[, j] * k) %*% KiZ)   
    }
  }
    
  if(grad){
    gr <- drop(crossprod(-2 * h * matrix(k, nrow(X), ncol(X)) , KiZ))
    return(list(gr = gr, Hes = res))
  }  
  return(res)
}
''')

grlaGP = robjects.r['dmean']
HelaGP = robjects.r['Hmean']

# # Tests
# model = laGP.laGP(Xref = np.atleast_2d(X[0,:]), start = 6, end = 12, X = X, Z = F[:,1], g = robjects.ListVector({'mle': True}), method = 'nn')
# pffla(np.atleast_2d(X[0,:]), model[6], model[7], 12, X,  F[:,1], meth = 'nn')
# numDeriv.grad(x = np.atleast_2d(X[0,:]), func = pffla, d = model[6], g = model[7], end = 12, Xtrain = X, Ftrain = F[:,1], methodL = 'nn')
# grlaGP(model, np.atleast_2d(X[0,:]), X[np.array(model[10]).astype('int') - 1,:],
#                           combine_fvals(F[np.array(model[10]).astype('int') - 1,:], 1))
# HelaGP(model, np.atleast_2d(X[0,:]), X[np.array(model[10]).astype('int') - 1,:],
#                           combine_fvals(F[np.array(model[10]).astype('int') - 1,:], 1))


def GPsurface_(X, Xtrain, Ftrain, upper, lower, covtype = None, numout=3, start=6, end=20, method='nn', sep=True,
               ginit=None, dinit=None, centerNorm=False, sd2=False):
    """Evaluates the RBF surface (and its gradient) at point(s) in X

    Parameters
    ----------
    X      = [numpoints-by-n] input vector to predict at
    Xtrain = [numpoints-by-n] input vector at training locations
    Ftrain      = [numpoints-by-ns] response matrix with replicates
    lower  = [dim] lower bound for lengthscale parameters
    upper  = [dim] upper ...
    covtype= [string5] covariance type, not used with laGP
    numout = [int] 1: just prediction, 2: add gradient, 3: add Hessian
    start = [int] number of initial starting designs
    end = [int] final number of points in the local design
    method = [string5] one of "alc", "alcopt", "alcray", "mspe", "nn", "fish", used to select
             local design points; see laGP documentation for more details
    sep = [bool] should a separable covariance be used (e.g., anisotropic)
    ginit = [RList] list to be passed to laGP about the signal to noise ratio / nugget parameter
    dinit = [RList] list to be passed to laGP about the lengthscale hyperparameters
    centerNorm = [bool] should F values be centered and normalized
    sd2 = [bool] to return predictive variance when numout = 1


    Returns
    ----------
    rbfvalue= [numpoints-by-1] ith entry is value of GP mean at row i of X
    Rbfgrad = [n-by-numpoints] ith column is gradient at row i of X
    RbfH = [n-by-n] Hessian at X
    dinit = [Rlist] list used by laGP about the signal to noise ratio / nugget parameter (unless more default laGP parameters are used)
    ginit = [Rlist] list used by laGP about the lengthscale hyperparameters (unless more default laGP parameters are used)
    gpsd2 = [numpoints-by-1] ith entry is value of GP variance at row i of X
    """
    X = np.atleast_2d(X)
    n = X.shape[1]
    ns = Ftrain.shape[1]

    end = min(Xtrain.shape[0] - 1, end)

    if(centerNorm):
        Fm = np.mean(Ftrain)
        Fv = np.var(combine_fvals(Ftrain, 1))
    else:
        Fm = 0
        Fv = 1

    Ftrain_sc = (Ftrain - Fm)/sqrt(Fv)

    # estimate signal to noise ratio
    signois = np.mean(np.var(Ftrain_sc, axis=1))/(sqrt(ns) * np.var(Ftrain_sc))

    if dinit is None:
        dinit = robjects.ListVector({'min': 1e-14})  # laGP default is too small
    if ginit is None:
        ginit = robjects.ListVector({'mle': True, 'start': max(1e-4, signois), 'min': 1.5e-8, 'max': max(1., signois)})

    if method == 'nn':
        start = end - 1  # faster

    # Build GP model (try estimating the hyperparameters with d_init, g_init; then just d)

    try:
        if sep and Xtrain.shape[0] > n + 2:
            model = laGP.laGPsep(Xref=X, start=start, end=end, X=Xtrain, Z=combine_fvals(Ftrain_sc, 1), d=dinit,
                                 g=ginit, method=method)
            dinit = robjects.ListVector({'mle': model.rx2('d').rx2('mle'), 'start': np.array(model.rx2('mle')).flatten()[0:n],
                                         'max': model.rx2('d').rx2('max'), 'min': model.rx2('d').rx2('min'),
                                         'ab': model.rx2('d').rx2('ab')})
            ginit = robjects.ListVector({'mle': model.rx2('g').rx2('mle'), 'start': model.rx2('mle').rx2('g'),
                                         'max': model.rx2('g').rx2('max'), 'min': model.rx2('g').rx2('min'),
                                         'ab': model.rx2('g').rx2('ab')})

        else:
            model = laGP.laGP(Xref=X, start=start, end=end, X=Xtrain, Z=combine_fvals(Ftrain_sc, 1), d=dinit,
                              g=ginit, method=method)
            dinit = robjects.ListVector({'mle': model.rx2('d').rx2('mle'), 'start': model.rx2('mle')[0],
                                         'max': model.rx2('d').rx2('max'), 'min': model.rx2('d').rx2('min'),
                                         'ab': model.rx2('d').rx2('ab')})
            ginit = robjects.ListVector({'mle': model.rx2('g').rx2('mle'), 'start': model.rx2('mle').rx2('g'),
                                         'max': model.rx2('g').rx2('max'), 'min': model.rx2('g').rx2('min'),
                                         'ab': model.rx2('g').rx2('ab')})

    except:
        model = laGP.laGP(Xref=X, start=start, end=end, X=Xtrain, Z=combine_fvals(Ftrain_sc, 1),
                          d=robjects.ListVector({'min': 1e-16}),
                          g=robjects.ListVector({'mle': True, 'min': 1e-4, 'max': 1e-1, 'start': 1e-2}), method=method)
        dinit = None
        ginit = None

    gpval = np.array(model[0])*sqrt(Fv) + Fm
    if sd2:
        gpsd2 = np.array(model[1])*Fv
    else:
        gpsd2 = None

    # Can be None?
    gpgrad = None
    gpH = None

    if numout == 2:
        predsgr1 = grlaGP(model, X, Xtrain[np.array(model[10]).astype('int') - 1,:],
                          combine_fvals(Ftrain_sc[np.array(model[10]).astype('int') - 1,:], 1))

        # predsgr1 = numDeriv.grad(x = X, func = pffla, d = model[6], g = model[7], end = end,
        #                          Xtrain = Xtrain, Ftrain = combine_fvals(Ftrain, 1), methodL = 'nn')
        gpgrad = np.array(predsgr1)*sqrt(Fv)
        gpgrad.flatten()


    if numout > 2:
        # predsgr1 = numDeriv.grad(x = X, func = pffla, d = model[6], g = model[7], end = end,
        #                          Xtrain = Xtrain, Ftrain = combine_fvals(Ftrain, 1), methodL = 'nn')
        # finitediffH = numDeriv.hessian(x = X, func = pffla, d = model[6], g = model[7], end = end,
        #                                Xtrain = Xtrain, Ftrain = combine_fvals(Ftrain, 1), methodL = 'nn')
        # gpgrad = np.array(predsgr1)
        # gpH = np.array(finitediffH)
        gpH = HelaGP(model, X, Xtrain[np.array(model[10]).astype('int') - 1,:],
                     combine_fvals(Ftrain_sc[np.array(model[10]).astype('int') - 1,:], 1), grad=True)

        gpH = np.array(gpH[1])*sqrt(Fv)
        gpgrad = np.array(gpH[0])*sqrt(Fv)
        gpgrad.flatten()

    return gpval, gpgrad, gpH, dinit, ginit, gpsd2

def combine_fvals(F, axis = None):
    return np.mean(F, axis=axis)


def minimize(f, x0, bounds, budget=10000, optin=None, **optkwds):
    ns = 10                  # number of local evals
    objfunc = OrbitObjectiveFunction(f, ns)

    n = len(x0)              # Number of variables
    rbftype = 'cubic'        # Type of RBF (multiquadric, cubic, Gaussian) ['cubic']
    nfmax = budget           # Maximum number of function evaluations per local minimization [60]
    nmpmax = 2*n + 1         # Maximum number of model points [2*n+1]
    trnorm = 0               # Type f trust-region norm [0]
    Low = bounds[:,0]        # 1-by-n Vector of lower bounds
    Upp = bounds[:,1]        # 1-by-n Vector of upper bounds
    gtol = 1e-05             # Gradient tolerance used to stop the local minimization [1e-5]
    gamma_m = 0.5            # Reduction factor = factor of the LHS points you'd start a local run from [.5]

    # laGP settings
    start = max(6, n+2)      # number of starting design (if not using method = 'nn'
    end = max(start + 1, 10*n)    # final number of local designs
    sep = True               # use anisotropic covariance?
    methodLa = 'alc'         # 'nn' for nearest neighbors (for speed), or, e.g., 'alc' (for accuracy)

    delta = 0.01             # initial trust region radius
    maxdelta = 0.5           # maximum trust region radius
    nfs = 8                  # number of initially evaluated points
    Xinit = x0[:]            # The points initially evaluated

    # generate some additional values (TODO: tune number)
    initials = list()
    for i in range(7):
        initials.append(Low + i*(Upp-Low)/7.)
    Xinit = np.vstack((Xinit, np.reshape(initials, [7, n])))
    F = np.apply_along_axis(objfunc, 1, Xinit)        # Their function values

    xkin = 0
    Xopt, F, xkin, nf, exitflag, xkin_mat, xkin_val = \
        ORBIT2(objfunc, rbftype, gamma_m, n, nfmax, nmpmax, delta, maxdelta,
               trnorm, gtol, Low, Upp, nfs, Xinit, F, xkin, ns, start=start, end=end,
               methodLa=methodLa, sep=sep)

    optval = min(F[xkin:][0])
    optpar = Xopt[xkin:][0]
    minval = min(F.flatten().tolist())

    # the following is debatable, but more in line with the other optimizers for a fair
    # comparison (Result should really take both minval and robust_minval)
    result = Result(minval, optpar)
    return result, objfunc.get_history()
