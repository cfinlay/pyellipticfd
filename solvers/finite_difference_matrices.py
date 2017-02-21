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
        Nint = N-2*w*Nx-2*w*(Ny-2*w)

        normv2 = v[0]**2 + v[1]**2
        vals = np.array([1, -2, 1])/(normv2*dx**2)

        shift = v[1] + Ny*v[0]

        I, J = np.indices((Nx,Ny))
        I = np.reshape(I,N)
        J = np.reshape(J,N)
        Ix = I*Ny+J
        Interior = np.logical_and(I >= w,
                        np.logical_and(I < Nx-w,
                            np.logical_and(J >=  w, J < Ny-w)))
        Ix_interior = Ix[Interior]


        R = np.tile(np.arange(Nint),3)
        C = np.concatenate([Ix_interior-shift, Ix_interior, Ix_interior+shift])
        data = np.repeat(vals,Nint)

        L = sparse.coo_matrix((data,(R,C)),shape=(Nint,N))
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
        Nint = N-2*w*Nx-2*w*(Ny-2*w)

        normv = np.linalg.norm(v)
        vals = np.array([-1, 1])/(normv*dx)

        shift = v[1] + Ny*v[0]

        I, J = np.indices((Nx,Ny))
        I = np.reshape(I,N)
        J = np.reshape(J,N)
        Ix = I*Ny+J
        Interior = np.logical_and(I >= w,
                        np.logical_and(I < Nx-w,
                            np.logical_and(J >=  w, J < Ny-w)))
        Ix_interior = Ix[Interior]


        R = np.tile(np.arange(Nint),2)
        C = np.concatenate([Ix_interior, Ix_interior+shift])
        data = np.repeat(vals,Nint)

        L = sparse.coo_matrix((data,(R,C)),shape=(Nint,N))
        Matrices.append(L.tocsr())

    return tuple(Matrices)
