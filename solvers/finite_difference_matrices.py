import numpy as np
import scipy.sparse as sparse
import itertools

def d2(shape,stencil,dx):
    """
    Return a list containing second order directional derivative 
    finite difference matrices for each stencil vector.
    """
    Nx = shape[0]
    Ny = shape[1]
    N = Nx*Ny

    Matrices = []
    for v in stencil:
        w = np.abs(v).max()

        normv2 = v[0]**2 + v[1]**2
        vals = np.array([1, -2, 1])/(normv2*dx**2)

        shift = v[1] + Ny*v[0]

        L = sparse.lil_matrix((N-2*w*Nx-2*w*(Ny-2*w), N))

        r = 0
        for i, j in itertools.product( range(0,Nx), range(0,Ny) ):
            s = i*Ny + j
            if i >= w and i < Nx-w and j >= w and j < Ny-w:
                if shift > 0:
                    L[r,[s-shift, s, s+shift]] = vals
                else:
                    L[r,[s+shift, s, s-shift]] = vals
                r = r+1
        Matrices.append(L.tocsr())

    return tuple(Matrices)

def d1(shape,stencil,dx):
    """
    Return a list containing firt order directional derivative 
    finite difference matrices for each stencil vector.
    """
    Nx = shape[0]
    Ny = shape[1]
    N = Nx*Ny

    Matrices = []
    for v in stencil:
        w = np.abs(v).max()

        normv = np.linalg.norm(v)
        vals = np.array([-1, 1])/(normv*dx)

        shift = v[1] + Ny*v[0]

        L = sparse.lil_matrix((N-2*w*Nx-2*w*(Ny-2*w), N))

        r = 0
        for i, j in itertools.product( range(0,Nx), range(0,Ny) ):
            s = i*Ny + j
            if i >= w and i < Nx-w and j >= w and j < Ny-w:
                if shift > 0:
                    L[r,[s, s+shift]] = vals
                else:
                    L[r,[s+shift, s]] = np.array([vals[1], vals[0]])
                r = r+1
        Matrices.append(L.tocsr())

    return tuple(Matrices)
