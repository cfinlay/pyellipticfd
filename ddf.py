"""Functions to calculate finite differences with Froese's method in 2D."""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from pyellipticfd import _ddutils, ddg

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

    # If paired antipodal points are available, then Froese's method
    # is equivalent to finite differences on a grid.
    if G.pairs is not None:
        return  ddg.d2(G,v,u=u,jacobian=jacobian)

    if G.dim !=2:
        raise TypeError('Dimensions other than two not supported.')

    # v must be an array of vectors, a direction for each interior point
    v = _ddutils.process_v(G,v)
    vth = np.arctan2(v[:,1], v[:,0]) # v's angle

    # Get finite difference simplices on interior
    mask = np.in1d(G.simplices[:,0], G.interior)
    interior_simplices = G.simplices[mask]
    I, S = interior_simplices[:,0], interior_simplices[:,1:]

    X = G.points[S] - G.points[I,None]  # The simplex vectors
    X = np.swapaxes(X,1,2)              # Transpose last two axis
    Xth = np.arctan2(X[:,1,:],X[:,0,:]) # angle of the simplex vectors

    # dictionary, to look up interior index from graph index
    d = dict(zip(G.interior,range(G.num_interior)))
    i = [d[key] for key in I]

    # Cone coordinates of direction to take derivative
    Xi = np.linalg.solve(X,v[i])

    mask_f = (Xi>=0).all(axis=1) # Farkas' lemma: in which simplex does the direction lie?
    mask_b = (Xi<=0).all(axis=1)

    # There could be multiple simplices for each direction. Choose one.
    _, If = np.unique(I[mask_f],return_index=True)
    _, Ib = np.unique(I[mask_b],return_index=True)

    # Forward and backward simplices
    Xf = X[mask_f][If]
    Xb = X[mask_b][Ib]

    # Concatenate them together
    X = np.concatenate([Xf, Xb],axis=2)
    h = np.linalg.norm(X,axis=1)

    # Angles for each stencil vector
    Xthf = Xth[mask_f][If]
    Xthb = Xth[mask_b][Ib]
    Xth = np.concatenate([Xthf, Xthb],axis=1)

    # Simplex indices
    Sf = S[mask_f][If]
    Sb = S[mask_b][If]
    S = np.concatenate([Sf,Sb],axis=1)

    # Angle with respect to direction vector
    dth = Xth - vth[:, None]
    dth %= 2*np.pi

    # Sort by quadrant
    jsort = np.argsort(dth)
    i = np.indices(dth.shape)[0]

    dth = dth[i,jsort]
    S = S[i,jsort]
    h = h[i,jsort]

    c = h*np.cos(dth)
    s = h*np.sin(dth)

    # It could be that the point in the fourth quadrant lies on the direction vector,
    # in which case the above would have sorted the points incorrectly.
    m = np.logical_and.reduce([c[:,0] >=0, c[:,1]>=0,
                               s[:,0] >=0, s[:,1]>=0],axis=0)
    c[m] = np.roll(c[m],-1,axis=1)
    s[m] = np.roll(s[m],-1,axis=1)
    S[m] = np.roll(S[m],-1,axis=1)

    denom = ((c[:,2]*s[:,1] - c[:,1]*s[:,2]) * (c[:,0]**2*s[:,3]-c[:,3]**2*s[:,0]) -
            (c[:,0]*s[:,3]-c[:,3]*s[:,0])*(c[:,2]**2*s[:,1] - c[:,1]**2*s[:,2]))
    a1 = 2*s[:,3]*(c[:,2]*s[:,1] - c[:,1]*s[:,2]) / denom
    a2 = 2*s[:,2]*(c[:,0]*s[:,3] - c[:,3]*s[:,0]) / denom
    a3 = -2*s[:,1]*(c[:,0]*s[:,3] - c[:,3]*s[:,0]) / denom
    a4 = -2*s[:,0]*(c[:,2]*s[:,1] - c[:,1]*s[:,2]) / denom
    a = np.stack([a1,a2,a3,a4],axis=1)

    if u is not None:
        d2u = np.sum(a*(u[S] - u[G.interior,None]),axis=1)
    else:
        d2u = None

    if jacobian:
        i = np.tile(np.repeat(G.interior,2*G.dim),2)
        j = np.concatenate([S.flatten(),np.repeat(G.interior,2*G.dim)])
        val = np.concatenate([a.flatten(),-a.flatten()])
        M = coo_matrix((val,(i,j)), shape = (G.num_interior,G.num_points)).tocsr()
    else:
        M = None

    return d2u, M

def _num_directions(G):
    """Number of search directions"""
    return np.max([int(np.ceil(2*np.pi / G.angular_resolution )),4])

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
        performance speed the next time d1grad is called.

    Returns
    -------
    lambda_min, lambda_max
        Respectively the minimal and maximal eigenvalues.
        These are tuples consisting of the value, Jacobian and control.
    """

    # If paired antipodal points are available, then Froese's method
    # is equivalent to finite differences on a grid.
    if G.pairs is not None:
        return ddg.d2eigs(G,u,jacobian=jacobian,control=control)

    if G.dim==3:
        raise NotImplementedError("Eigenvalues not implemented in 3d.")

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
        val = np.zeros(5*G.num_interior)
        col = np.zeros(5*G.num_interior,dtype=np.intp)
        row = np.zeros(5*G.num_interior,dtype=np.intp)

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

        val = np.zeros(5*G.num_interior)
        col = np.zeros(5*G.num_interior,dtype=np.intp)
        row = np.zeros(5*G.num_interior,dtype=np.intp)

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
