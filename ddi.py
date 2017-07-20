"""Functions to calculate finite differences with linear interpolation."""

import warnings
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

from pyellipticfd import _ddutils

# TODO fix this eps hack
eps = 1e-15

def d1(G,v, u=None, jacobian=False, domain="interior"):
    """
    Compute the directional derivative of u, in direction v.

    Parameters
    ----------
    G : FDPointCloud
        The mesh of grid points.
    v : array_like
        Direction to take firt derivative.
    u : array_like
        Function values at grid points. If not specified, only the Jacobian
        is calculated.
    jacobian : boolean
        Whether to also calculate the Jacobian M.
    domain : string
        Which points to compute derivative on: one of "interior",
        "boundary". If not specified, defaults to "interior".

    Returns
    -------
    d1u : array_like
        First derivative in the direction v.
    M : scipy csr_matrix
        Finite difference matrix.
    """

    if u is None:
        jacobian = True

    # v must be an array of vectors, a direction for each point
    v = _ddutils.process_v(G,v,domain=domain)

    if domain=="interior":
        Ix = G.interior
    elif domain=="boundary":
        Ix = G.bdry

    # Get finite difference simplices on correct domain
    mask = np.in1d(G.simplices[:,0], Ix)
    simplices = G.simplices[mask]
    I, S = simplices[:,0], simplices[:,1:]

    X = G.points[S] - G.points[I,None] # The simplex vectors
    X = np.swapaxes(X,1,2)             # Transpose last two axis

    # dictionary, to look up domain index from graph index
    d = dict(zip(Ix,range(Ix.size)))
    i = [d[key] for key in I]

    # Cone coordinates of direction to take derivative
    Xi = np.linalg.solve(X,v[i])

    if domain=="interior":
        mask_f = (Xi>=-eps/G.spatial_resolution).all(axis=1)
    elif domain=="boundary":
        mask = np.logical_and(Xi>=-eps/G.spatial_resolution, np.reshape(np.in1d(S,G.interior),S.shape))
        mask_f = mask.all(axis=1)

    _, If = np.unique(I[mask_f],return_index=True)

    Sf = S[mask_f][If]

    Xif = Xi[mask_f][If]

    if u is not None:
        d1u = np.einsum('ij,ij->i',u[Sf] - u[Ix,None],Xif)
    else:
        d1u = None

    if jacobian:
        i = np.tile(np.repeat(Ix,G.dim),2)
        j = np.concatenate([Sf.flatten(), np.repeat(Ix,G.dim)])
        val = np.concatenate([Xif.flatten(),-Xif.flatten()])
        M = coo_matrix((val,(i,j)), shape = [G.num_points]*2).tocsr()
        M = M[M.getnnz(1)>0]
    else:
        M = None

    return d1u, M


def d1n(G,u=None,**kwargs):
    """
    Compute the directional derivative with respect to the outward normal direction
    of the boundary of the domain. If no function is given, theni only the Jacobian
    is calculated.

    Parameters
    ----------
    G : FDPointCloud
        The mesh of grid points.
    u : array_like
        Function values at grid points. If not specified, only the Jacobian
        is calculated.
    jacobian : boolean
        Whether to also calculate the Jacobian M.

    Returns
    -------
    d1u : array_like
        First derivative in the direction v.
    M : scipy csr_matrix
        Finite difference matrix.
    """
    return d1(G, -G.bdry_normals, u=u, domain='boundary', **kwargs)


def _num_directions(G):
    """Number of search directions"""
    return np.max([int(2*np.ceil(np.pi**2 / G.angular_resolution**2 )),4])

def d1grad(G,u,jacobian=False,control=False, cache_fdmatrices=True):
    """
    Compute the first derivative of U in the direction of the gradient,
    on the interior of the domain.

    Parameters
    ----------
    G : FDPointCloud
        The mesh of grid points.
    u : array_like
        Function values at grid points.
    jacobian : boolean
        If True, also calculate the Jacobian (finite difference matrix).
    control : boolean
        If True, also calculate the gradient direction.
    cache_fdmatrices : boolean
        If True, then the finite difference matrices used here are cached
        in the FDPointCloud. Caching these matrices drastically improves
        performance speed the next time d1grad is called.

    Returns
    -------
    d1_max : array_like
        The value of the first derivative of U in the gradient direction.
    M : scipy csr_matrix
        Finite difference matrix.
    v : array_like
        The gradient direction.
    """

    if G.dim==3:
        raise NotImplementedError("Gradient not yet implemented in 3d.")

    cache_exists = G._d1cache is not None
    if cache_exists:
        if G._d1cache[0]!=__name__:
            warnings.warn('Cache exists but not created by ', __name__, '. Overwriting.')
            cache_fdmatrices=True
            cache_exist=False

    if not cache_exists:
        # number of directions to take
        Nds = _num_directions(G)
        th = np.arange(0,2*np.pi,2*np.pi/Nds)
        V = np.stack([np.cos(th),np.sin(th)],axis=1)

        #Take directional derivatives
        get_fdms = cache_fdmatrices or jacobian
        d1tup = [d1(G,v,u,jacobian=get_fdms) for v in V]

        d1u = np.stack([tup[0] for tup in d1tup],axis=1)

        # Finite difference matrices, for each direction
        if get_fdms:
            fdms = [tup[1] for tup in d1tup]

        if cache_fdmatrices:
            G._d1cache = (__name__,V,fdms)
    else:
        V = G._d1cache[1]
        fdms = G._d1cache[2]

        d1u = np.stack([M.dot(u) for M in fdms], axis=1)

    ixmax = d1u.argmax(axis=1)
    i = G.interior
    d1_max = d1u[i,ixmax]

    if control:
        v = V[ixmax]
    else:
        v = None

    if jacobian:
        val = np.zeros((G.dim+1)*G.num_interior)
        col = np.zeros((G.dim+1)*G.num_interior,dtype=np.intp)
        row = np.zeros((G.dim+1)*G.num_interior,dtype=np.intp)

        count = 0
        for r, ix in enumerate(ixmax):
            Mr = fdms[ix][r].tocoo()
            n = Mr.data.size

            val[count:(count+n)] = Mr.data
            col[count:(count+n)] = Mr.col
            row[count:(count+n)] = np.full(Mr.data.size,r)

            count +=n



        M = csr_matrix((val[:count], (row[:count], col[:count])),
                        shape=(G.num_interior, G.num_points))
    else:
        M = None


    return d1_max, M, v

def d2(G,v,u=None,jacobian=False):
    """
    Compute the second directional derivative of u, in direction v.

    Parameters
    ----------
    G : FDGraph
        The mesh of grid points.
    v : array_like
        Direction to take second derivative.
    u : array_like
        Function values at grid points. If not specified, only the Jacobian
        is calculated.
    jacobian : boolean
        Whether to also calculate the Jacobian M.

    Returns
    -------
    d2u : array_like
        Second derivative value in the direction v.
    M : scipy csr_matrix
        Finite difference matrix.
    """
    if u is None:
        jacobian = True

    # v must be an array of vectors, a direction for each interior point
    v = _ddutils.process_v(G,v)

    # Get finite difference simplices on interior
    mask = np.in1d(G.simplices[:,0], G.interior)
    interior_simplices = G.simplices[mask]
    I, S = interior_simplices[:,0], interior_simplices[:,1:]

    X = G.points[S] - G.points[I,None] # The simplex vectors
    X = np.swapaxes(X,1,2)             # Transpose last two axis

    # dictionary, to lookup interior index from graph index
    d = dict(zip(G.interior,range(G.num_interior)))
    i = [d[key] for key in I]

    # Cone coordinates of direction to take derivative
    Xi = np.linalg.solve(X,v[i])

    mask_f = (Xi>=-eps/G.spatial_resolution).all(axis=1)# Farkas' lemma: in which simplex does the direction lie
    mask_b = (Xi<=eps/G.spatial_resolution).all(axis=1)

    _, If = np.unique(I[mask_f],return_index=True)
    _, Ib = np.unique(I[mask_b],return_index=True)

    Sf = S[mask_f][If]
    Sb = S[mask_b][If]

    Xif = Xi[mask_f][If]
    Xib = -Xi[mask_b][Ib]

    hf = 1/np.sum(Xif,axis=1)
    hb = 1/np.sum(Xib,axis=1)

    if u is not None:
        dUf = np.einsum('ij,ij->i',u[Sf] - u[G.interior,None],Xif)
        dUb = np.einsum('ij,ij->i',u[Sb] - u[G.interior,None],Xib)

        d2u = 2/(hf+hb)*(dUf + dUb)
    else:
        d2u = None

    if jacobian:
        i = np.tile(np.repeat(G.interior,G.dim),4)
        j = np.concatenate([Sf.flatten(),np.repeat(G.interior,G.dim),
                            Sb.flatten(),np.repeat(G.interior,G.dim)])
        val = np.concatenate([Xif.flatten(),-Xif.flatten(),
                              Xib.flatten(), -Xib.flatten()])
        M = coo_matrix((val,(i,j)), shape = (G.num_interior,G.num_points)).tocsr()
        M = diags(2/(hf+hb),format="csr").dot(M)
    else:
        M = None

    return d2u, M


def d2eigs(G,u,jacobian=False, control=False, cache_fdmatrices=True):
    """
    Compute the eigenvalues of the Hessian of U.

    Parameters
    ----------
    G : FDPointCloud
        The mesh of grid points.
    u : array_like
        Function values at grid points.
    jacobian : boolean
        Whether to calculate the Jacobians for each eigenvalue.
    control : boolean
        Whether to calculate the directions of the eigenvalues.
    cache_fdmatrices : boolean
        If True, then the finite difference matrices used here are cached
        in the FDPointCloud. Caching these matrices drastically improves
        performance speed the next time d2eigs is called.

    Returns
    -------
    lambda_min, lambda_max
        Respectively the minimal and maximal eigenvalues.
        These are tuples consisting of the value, Jacobian and control.
    """

    if G.dim==3:
        raise NotImplementedError("Eigenvalues not yet implemented in 3d.")

    cache_exists = G._d2cache is not None
    if cache_exists:
        if G._d2cache[0]!=__name__:
            warnings.warn('Cache exists but not created by ', __name__, '. Overwriting.')
            cache_fdmatrices=True
            cache_exist=False

    if not cache_exists:
        Nds = _num_directions(G)
        th = np.arange(0,2*np.pi,2*np.pi/Nds)
        V = np.stack([np.cos(th),np.sin(th)],axis=1)

        #Take directional derivatives
        get_fdms = cache_fdmatrices or jacobian
        d2tup = [d2(G,v,u,jacobian=get_fdms) for v in V]

        d2u = np.stack([tup[0] for tup in d2tup],axis=1)

        # Finite difference matrices, for each direction
        if get_fdms:
            fdms = [tup[1] for tup in d2tup]

        if cache_fdmatrices:
            G._d2cache = (__name__,V,fdms)
    else:
        V = G._d2cache[1]
        fdms = G._d2cache[2]

        d2u = np.stack([M.dot(u) for M in fdms], axis=1)

    arg = np.argsort(d2u)
    ixvmin, ixvmax = arg[:,0], arg[:,-1]
    i = G.interior
    d2min = d2u[i,ixvmin]
    d2max = d2u[i,ixvmax]

    if jacobian:
        val = np.zeros((2*G.dim+1)*G.num_interior)
        col = np.zeros((2*G.dim+1)*G.num_interior,dtype=np.intp)
        row = np.zeros((2*G.dim+1)*G.num_interior,dtype=np.intp)

        count = 0
        for r, ix in enumerate(ixvmin):
            Mr = fdms[ix][r].tocoo()
            n = Mr.data.size

            val[count:(count+n)] = Mr.data
            col[count:(count+n)] = Mr.col
            row[count:(count+n)] = np.full(Mr.data.size,r)

            count +=n

        M_min = csr_matrix((val[:count], (row[:count], col[:count])),
                        shape=(G.num_interior, G.num_points))

        val = np.zeros((2*G.dim+1)*G.num_interior)
        col = np.zeros((2*G.dim+1)*G.num_interior,dtype=np.intp)
        row = np.zeros((2*G.dim+1)*G.num_interior,dtype=np.intp)

        count = 0
        for r, ix in enumerate(ixvmax):
            Mr = fdms[ix][r].tocoo()
            n = Mr.data.size

            val[count:(count+n)] = Mr.data
            col[count:(count+n)] = Mr.col
            row[count:(count+n)] = np.full(Mr.data.size,r)

            count +=n

        M_max = csr_matrix((val[:count], (row[:count], col[:count])),
                        shape=(G.num_interior, G.num_points))

    else:
        M_min, M_max = [None]*2

    if control:
        v_min, v_max = V[ixvmin], V[ixvmax]
    else:
        v_min, v_max = [None]*2

    return (d2min, M_min, v_min), (d2max, M_max, v_max)


def d2min(G,u,**kwargs):
    """
    Compute the minimum eigenvalues of the Hessian of u.
    """
    return d2eigs(G,u,**kwargs)[0]

def d2max(G,u,**kwargs):
    """
    Compute the maximum eigenvalues of the Hessian of u.
    """
    return d2eigs(G,u,**kwargs)[1]
