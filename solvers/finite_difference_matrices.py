import numpy as np
import scipy.sparse as sparse
import itertools

def domain_indices(shape,w):
    """
    Return the interior and exterior unzipped indices on a domain with boundary width w.
    """
    N = np.prod(shape)
    Nx = shape[0]
    Ny = shape[1]

    I, J = np.indices(shape)
    I = np.reshape(I,N)
    J = np.reshape(J,N)
    Ix = I*Ny+J
    Interior = np.logical_and(I >= w,
                    np.logical_and(I < Nx-w,
                        np.logical_and(J >=  w, J < Ny-w)))
    Ix_interior = Ix[Interior]
    Exterior = np.logical_not(Interior)
    Ix_exterior = Ix[Exterior]

    return Ix_interior, Ix_exterior


def d2(shape,stencil,dx,ix=None, boundary=False):
    """
    If no stencil indices are given, return a tuple of second order
    directional derivative finite difference matrices for each stencil vector.

    Otherwise return a FD matrix for the unzipped vector, taking second
    order differences along the stencil vector specified by the index.
    """
    Nx = shape[0]
    Ny = shape[1]
    N = Nx*Ny

    if ix is None:
        Matrices = []
        for v in stencil:
            w = np.abs(v).max()
            Nint = N-2*w*Nx-2*w*(Ny-2*w)

            normv2 = v[0]**2 + v[1]**2
            vals = np.array([1, -2, 1])/(normv2*dx**2)

            shift = v[1] + Ny*v[0]

            Ix_interior = domain_indices(shape,w)[0]


            R = np.tile(np.arange(Nint),3)
            C = np.concatenate([Ix_interior-shift, Ix_interior, Ix_interior+shift])
            data = np.repeat(vals,Nint)

            L = sparse.coo_matrix((data,(R,C)),shape=(Nint,N))
            Matrices.append(L.tocsr())

        return tuple(Matrices)
    else:
        w = np.abs(stencil).max(1).min()
        Nint = N-2*w*Nx-2*w*(Ny-2*w)

        ix = np.reshape(ix,ix.size)

        v = stencil[ix]
        normv2 = v[:,0]**2 + v[:,1]**2

        vals = np.repeat([1,-2,1],Nint)/np.tile(normv2*dx**2,3)

        shift = v[:,1] + Ny*v[:,0]

        Ix_interior, Ix_exterior = domain_indices(shape,w)

        if not boundary:
            R = np.tile(np.arange(Nint),3)
            C = np.concatenate([Ix_interior-shift, Ix_interior, Ix_interior+shift])

            L = sparse.coo_matrix((vals,(R,C)),shape=(Nint,N))
        else:
            R = np.concatenate([np.tile(np.arange(Nint),3),np.arange(Nint,N)])
            C = np.concatenate([Ix_interior-shift, Ix_interior, Ix_interior+shift, Ix_exterior])

            L = sparse.coo_matrix((np.concatenate([vals,np.ones(N-Nint)]),
                                  (R,C)),shape=(N,N))

        return L.tocsr()



def d1(shape,stencil,dx,ix=None, boundary=False):
    """
    If no stencil indices are given, return a tuple of first order
    directional derivative finite difference matrices for each stencil vector.

    Otherwise return a FD matrix for the unzipped vector, taking first
    order differences along the stencil vector specified by the index.
    """
    Nx = shape[0]
    Ny = shape[1]
    N = Nx*Ny

    if ix is None:
        Matrices = []
        for v in stencil:
            w = np.abs(v).max()
            Nint = N-2*w*Nx-2*w*(Ny-2*w)

            normv = np.linalg.norm(v)
            vals = np.array([-1, 1])/(normv*dx)

            shift = v[1] + Ny*v[0]

            Ix_interior = domain_indices(shape,w)[0]


            R = np.tile(np.arange(Nint),2)
            C = np.concatenate([Ix_interior, Ix_interior+shift])
            data = np.repeat(vals,Nint)

            L = sparse.coo_matrix((data,(R,C)),shape=(Nint,N))
            Matrices.append(L.tocsr())

        return tuple(Matrices)
    else:
        w = np.abs(stencil).max(1).min()
        Nint = N-2*w*Nx-2*w*(Ny-2*w)

        ix = np.reshape(ix,ix.size)

        v = stencil[ix]
        normv = np.linalg.norm(v,axis=1)

        vals = np.repeat([-1,1],Nint)/np.tile(normv2*dx**2,2)

        shift = v[:,1] + Ny*v[:,0]

        Ix_interior, Ix_exterior = domain_indices(shape,w)

        if not boundary:
            R = np.tile(np.arange(Nint),2)
            C = np.concatenate([Ix_interior, Ix_interior+shift])

            L = sparse.coo_matrix((vals,(R,C)),shape=(Nint,N))
        else:
            Exterior = np.logical_not(Interior)
            Ix_exterior = Ix[Exterior]

            R = np.concatenate([np.tile(np.arange(Nint),2),np.arange(Nint,N)])
            C = np.concatenate([Ix_interior, Ix_interior+shift, Ix_exterior])

            L = sparse.coo_matrix((np.concatenate([vals,np.ones(N-Nint)]),
                                  (R,C)),shape=(N,N))

        return L.tocsr()
