"""Functions to create finite difference stencils on regular grids."""

import numpy as np
import math

def stern_brocot(N):
    """
    Generate a Stern Brocot tree of depth N.
    """
    sb_old = np.array([[1,0],
                     [1,1],
                     [0,1]],dtype=np.intp)

    if N==1:
        return sb_old

    for n in range(1,N):
        sb_new = [np.array([v,v+w]) for (v,w) in zip(sb_old[:-1], sb_old[1:])]
        sb_new.append([sb_old[-1]])
        sb_old = np.concatenate(sb_new)

    return sb_old

# Stencil vectors
def create_2d(r):
    """
    Create a 2D stencil with radius r.
    """
    if r<1:
        raise ValueError('r must be positive')

    theta = np.pi/2
    R = np.array([[0, -1],
                  [1,  0]],dtype=np.intp)


    V = stern_brocot(r)[0:-1]
    V = V[(V<=r).all(1),:].T

    stencils = [np.linalg.matrix_power(R,i).dot(V).T for i in range(0,4)]

    return np.concatenate(stencils)

gcd = np.frompyfunc(math.gcd, 2, 1)

def create_nd(r,n):
    """
    Create an n dimensional stencil with radius r.
    """
    if r<1:
        raise ValueError('r must be positive')

    shape = (2*r+1 for d in range(n))
    stencil = np.indices(shape)-r

    stencil = stencil.reshape((n,(2*r+1)**n)).T
    stencil = stencil[(stencil != 0).any(axis=1),:]

    g = gcd.reduce(stencil, axis=1)

    return stencil[g==1,:]

def create(r,d):
    """
    Create a d dimensional stencil with radius r.
    """
    if d<2:
        raise ValueError('dimension must be two or higher')
    elif d==2:
        return create_2d(r)
    else:
        return create_nd(r,d)
