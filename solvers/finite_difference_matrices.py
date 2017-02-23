import numpy as np
import scipy.sparse as sparse
import itertools

def d2(shape,stencil,dx,ix=None):
    """
    If no stencil indices are given, return a tuple of second order
    directional derivative finite difference matrices for each stencil vector.

    Otherwise return a FD matrix for the unzipped vector, taking second
    order differences along the stencil vector specified by the index.
    """
    Nx = shape[0]
    Ny = shape[1]
    N = Nx*Ny

    I,J = np.indices(shape)

    if ix is None:
        Matrices = []
        for v in stencil:
            w = np.abs(v).max()
            Nint = N-2*w*Nx-2*w*(Ny-2*w)

            normv2 = v[0]**2 + v[1]**2
            vals = np.array([1, -2, 1])/(normv2*dx**2)

            shift = v[1] + Ny*v[0]

            Ix_interior = np.ravel_multi_index((I[w:-w,w:-w],J[w:-w,w:-w]),shape)
            Ix_interior = Ix_interior.reshape(Ix_interior.size)


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

        Ix_interior = np.ravel_multi_index((I[w:-w,w:-w],J[w:-w,w:-w]),shape)
        Ix_interior = Ix_interior.reshape(Ix_interior.size)

        R = np.tile(np.arange(Nint),3)
        C = np.concatenate([Ix_interior-shift, Ix_interior, Ix_interior+shift])

        L = sparse.coo_matrix((vals,(R,C)),shape=(Nint,N))

        return L.tocsr()



def d1(shape,stencil,dx,ix=None):
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

            Ix_interior = np.ravel_multi_index((I[w:-w,w:-w],J[w:-w,w:-w]),shape)
            Ix_interior = Ix_interior.reshape(Ix_interior.size)


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

        Ix_interior = np.ravel_multi_index((I[w:-w,w:-w],J[w:-w,w:-w]),shape)
        Ix_interior = Ix_interior.reshape(Ix_interior.size)

        R = np.tile(np.arange(Nint),2)
        C = np.concatenate([Ix_interior, Ix_interior+shift])

        L = sparse.coo_matrix((vals,(R,C)),shape=(Nint,N))
        return L.tocsr()
