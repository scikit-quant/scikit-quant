from __future__ import print_function
"""
Supplementary or Auxiliary Functions for Snobfit

Most of them are the functions of the Rough MATLAB-Numpy Equivalents.
"""

import numpy

# Some constants
eps   = numpy.finfo('d').eps
inf   = numpy.inf
nan   = numpy.nan

# Some internal functions of np
norm  = numpy.linalg.norm
inv   = numpy.linalg.inv
triu  = numpy.triu
isnan = numpy.isnan


# ----------------------------------------------------------
def diag(arr):
    return numpy.diag(arr.flatten())

def within(low, x, high):
    return numpy.logical_and(low <= x, high >= x)


def feval(funcName, *args):
     return eval(funcName)(*args)


def find(cond_array):
    #cond_array should already have np conditional applied
    # tup = numpy.where(*args,**kw)
    # if len(tup)==0:
    #     return numpy.array([],'int')
    # else:
    #     return tup[0]
    return (numpy.transpose(numpy.nonzero(cond_array.flatten()))).astype(int)
    # return numpy.array([i for (i, val) in enumerate(a) if func(val)])


def toCol(v):
    """
    v is a vector
    """
    return numpy.matrix(v.reshape(v.size, 1))


def toRow( m ):
    """
    m is a  n x 1 matrix
    """
    n = m.shape[0]
    return numpy.array( numpy.asarray(m).reshape(1,n)[0] )
    # numpy.array( m.A.reshape(1,n)[0] )


def vector(n):
     """
     Return a vector of the given length.
     """
     return numpy.empty(n,'d')

def ivector(n=0):
    """
    Return a int vector of the given length.
    """
    return numpy.zeros(n,'int')


def sort(x):
    ind = numpy.argsort(x)
    return x[ind], ind


def std(x):
    """
    STD(X) normalizes by (N-1) where N is the sequence length.
    This makes STD(X).^2 the best unbiased estimate of the variance
    if X is a sample from a normal distribution.
    """
    return numpy.std(x,ddof=1)


def max_(x):
    """
    (Y,I) = _max(x) returns the index I of the maximum values in vector x.
    If the values along the first non-singleton dimension contain more
    than one maximal element, the index of the first one is returned.
    """
    if x.size <= 0:
        return (numpy.array([None]), numpy.array([None]))
    idx = numpy.argmax(x)
    return x[idx], idx


def min_(x):
    """
    [Y,I] = MIN(X) returns the index of the minimum value in vector I.
    If the values along the first non-singleton dimension contain more
    than one minimal element, the index of the first one is returned.
    """
    if x.size <= 0:
        return (numpy.array([None]), numpy.array([None]))
    idx = numpy.argmin(x)
    return x[idx], idx

def maximum_(a, b):
    return numpy.max(numpy.concatenate((a, b)))

def extend(f, n):
    f = numpy.append(f, numpy.zeros((f.shape[0], n)), axis=1)
    return f


def toInt(x):
    # Make sure it is vector
    if x.shape[0]==1:
        x = numpy.reshape( numpy.asarray(x), x.shape[1] )

    Ix =  numpy.array( x, 'int' )
    # 10000 is big enough??? Fix me later.

    Ix[ numpy.where(Ix<0) ] = 10000
    return Ix


def isEmpty(x):
    if len(x)==0: return True
    if x.shape[0]==0: return True
    if x.shape[1]==0: return True
    return False


def removeByInd(a, ind):
    n = len(a) - len(ind)
    ret = numpy.zeros( n, 'int' )
    k = 0
    for i in xrange(len(a)):
        if (i not in ind):
            ret[k] = a[i]
            k += 1
    return ret


def duplicate(v, n, axis=0):
    #return np matrix
    return numpy.mat( v ).repeat(n, axis=axis)


def dup(v, n, axis=0):
    # Return a np array
    if axis == 0:
        return numpy.array( [v] ).repeat(n, axis=axis)
    else:
        return numpy.array( v ).repeat(n, axis=axis)


def dot3(a,b,c):
    # Calculate a*(b*c)
    return numpy.dot(a, numpy.dot(b,c) )


def crdot(m, v):
    n = len(v)
    _sum = 0.0
    for i in xrange(n):
        _sum += m[i,0]*v[i]

    return _sum


#---------------------------------------------------------------
_randstate = numpy.random.RandomState(6)
def rand(*args):
    """
    For debugging purposes, select an initial state that can be
    reproduced with octave:

    %%%%%
    % https://stackoverflow.com/questions/13735096/python-vs-octave-random-generator/13876670#13876670
    function state = mtstate(seed)

    state = uint32(zeros(625,1));

    state(1) = uint32(seed);
    for i=1:623,
       tmp = uint64(1812433253)*uint64(bitxor(state(i),bitshift(state(i),-30)))+i;
       state(i+1) = uint32(bitand(tmp,uint64(intmax('uint32'))));
    end
    state(625) = 1;

    % initialize
    rand('state', mtstate(4));
    %%%%%

    Further, when asking for multi-dimensional matrices of random numbers,
    octave and numpy order things differently, so fix that, too.
    """

    if len(args) == 1:
        return _randstate.rand(*args)
    elif len(args) == 2:
        res = numpy.zeros(args)
        for i in range(args[1]):
            res[:,i] = _randstate.rand(args[0])
        return res
    else:
        # more dims not needed (yet); deal with it if we get there ...
        raise NotImplementedError("no implementation for %d-dim" % len(args))

    res = numpy.zeros(args)
    idim = 0
    for arg in args:
        res[:,idim] = _randstate.rand(arg)
        idim += 1
    return res

#---------------------------------------------------------------
def rsort(x, w=None, remove_dups=True):
    """
    Sort x in increasing order, remove multiple entries,
    and adapt weights w accordingly x and w must both be a row or a column
    default input weights are w=1

    If w==None, the weighted empirical cdf is computed at x
    dof = len(x) at in

    Warning: when you use this function, make sure x and w is row vector
    """
    if  w is None:
        w = numpy.ones(x.shape)

    ind = numpy.argsort(x)
    x = x[ind]
    w = w[ind]

    n = len(x)

    # Remove dubplicates if requested
    xnew = numpy.append(x[1:n], inf)
    if remove_dups:
        ind = numpy.nonzero(xnew != x)
    else:
        ind = numpy.nonzero(range(1, len(x)+1))
    nn  = len(ind)
    x = x[ind]

    ww = numpy.zeros(nn)
    if nn>1:
        ww[0] = sum(w[ind[0]+1])
        for i in range(nn-1):
            ww[i+1] = sum(w[ind[i:i+1]])

    # Get cumulative sum of weights
    cdfx = numpy.cumsum(ww)

    # Adjust for jumps and normalize
    if cdfx[nn-1] != 0.:
        cdfx = (cdfx-0.5*ww)/cdfx[nn-1]
    dof = n

    return x, ww, cdfx, dof
