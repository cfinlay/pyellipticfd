import numpy as np
import itertools

def stern_brocot(N):
    """
    Generate a Stern Brocot tree of depth N.
    """
    sold = np.array([[1,0],
                     [1,1],
                     [0,1]],dtype=np.intp)

    if N==1:
        return sold

    n=1
    while n<N:
        snew = np.zeros((2*sold.shape[0]-1,2),dtype=np.intp)
        k = 0
        for i in range(sold.shape[0]-1):
            snew[k] = sold[i]
            snew[k+1] = sold[i] + sold[i+1]
            k+=2
        snew[k]=sold[i+1]
        sold=snew
        n+=1
    return snew

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


    V = stern_brocot(r)[0:-1].T

    stencils = []
    for i in range(0,4):
        stencils.append( (np.linalg.matrix_power(R,i)).dot(V).T)

    return np.concatenate(stencils)

def __gcd(a, b):
    a, b = np.broadcast_arrays(a, b)
    a = a.copy()
    b = b.copy()
    pos = np.nonzero(b)[0]
    while len(pos) > 0:
        b2 = b[pos]
        a[pos], b[pos] = b2, a[pos] % b2
        pos = pos[b[pos]!=0]
    return a

def create_3d(r):
    """
    Create a 3D stencil with radius r.
    """
    if r<1:
        raise ValueError('r must be positive')

    stencil = np.zeros(((2*r+1)**3,3),dtype=np.intp)
    ran = range(-r,r+1)
    for l,(i,j,k) in enumerate(itertools.product(ran,ran,ran)):
        stencil[l] = np.array([i,j,k])

    stencil = stencil[(stencil != 0).any(axis=1),:]

    gcd_ij = __gcd(stencil[:,0],stencil[:,1])
    gcd_k = __gcd(gcd_ij,stencil[:,2])

    stencil= stencil[np.abs(gcd_k)==1,:]
    return stencil

def create(r,d):
    """
    Create a d dimensional stencil with radius r.
    """
    if d==2:
        return create_2d(r)
    elif d==3:
        return create_3d(r)
    else:
        raise ValueError('Only dimensions 2 & 3 supported')
