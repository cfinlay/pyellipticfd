"""Functions to calculate finite differences with Froese's method in 2D."""

import numpy as np
from scipy.sparse import csr_matrix

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

    X = G.points[S] - G.points[I,None] # The simplex vectors
    X = np.swapaxes(X,1,2)                 # Transpose last two axis
    Xth = np.arctan2(X[:,1,:],X[:,0,:])

    # dictionary, to look up interior index from graph index
    d = dict(zip(G.interior,range(G.num_interior)))
    i = [d[key] for key in I]

    # Cone coordinates of direction to take derivative
    Xi = np.linalg.solve(X,v[i])
    vth = vth[i]

    # Given interior index, compute directional derivative
    def d2(k):
        mask = I==k
        xi = Xi[mask] # cone coordinates
        x = X[mask]   # stencil vectors
        s = interior_simplices[mask] # simplex indices

        th = vth[mask]
        xth = Xth[mask]
        dth = xth - th[:,None]

        # Forward direction
        mask_f = np.squeeze((xi>=0).all(axis=1)) # Farkas' lemma
        xth_f = xth[mask_f][0]
        th_f = th[mask_f][0]
        dth_f = dth[mask_f][0]
        m = dth_f < -np.pi
        dth_f[m] = dth_f[m]+2*np.pi
        x_ = x[mask_f][0]
        h_ = np.linalg.norm(x_,axis=0)
        s_ = s[mask_f][0][1:]
        if (dth_f!=0).all():
            h1, h4 = h_[dth_f>0], h_[dth_f<0]
            th1, th4 = dth_f[dth_f>0], dth_f[dth_f<0]
            j1,j4 = s_[dth_f>0], s_[dth_f<0]
        elif (dth_f>=0).all():
            h1, h4 = h_[dth_f>0], h_[dth_f==0]
            th1, th4 = dth_f[dth_f>0], dth_f[dth_f==0]
            j1,j4 = s_[dth_f>0], s_[dth_f==0]
        elif (dth_f<=0).all():
            h1, h4 = h_[dth_f==0], h_[dth_f<0]
            th1, th4 = dth_f[dth_f==0], dth_f[dth_f<0]
            j1,j4 = s_[dth_f==0], s_[dth_f<0]

        # Backward direction
        mask_b = np.squeeze((xi<=0).all(axis=1)) # Farkas' lemma
        xth_b = xth[mask_b][0]
        xth_b %= 2*np.pi
        th_b = th[mask_b][0]+np.pi
        th_b %= 2*np.pi
        Bdth_b = xth_b - th_b
        m = Bdth_b >np.pi
        Bdth_b[m] = Bdth_b[m] - 2*np.pi

        xth_b = xth[mask_b][0]
        xth_b %= 2*np.pi
        th_b = th[mask_b][0]
        th_b %= 2*np.pi
        dth_b = xth_b - th_b

        x_ = x[mask_b][0]
        h_ = np.linalg.norm(x_,axis=0)
        s_ = s[mask_b][0][1:]
        if (Bdth_b!=0).all():
            h3, h2 = h_[Bdth_b>0], h_[Bdth_b<0]
            th3, th2 = dth_b[Bdth_b>0], dth_b[Bdth_b<0]
            j3,j2 = s_[Bdth_b>0], s_[Bdth_b<0]
        elif (Bdth_b>=0).all():
            h3, h2 = h_[Bdth_b>0], h_[Bdth_b==0]
            th3, th2 = dth_b[Bdth_b>0], dth_b[Bdth_b==0]
            j3,j2 = s_[Bdth_b>0], s_[Bdth_b==0]
        elif (Bdth_b<=0).all():
            h3, h2 = h_[Bdth_b==0], h_[Bdth_b<0]
            th3, th2 = dth_b[Bdth_b==0], dth_b[Bdth_b<0]
            j3,j2 = s_[Bdth_b==0], s_[Bdth_b<0]

        h = np.squeeze(np.array([h1,h2,h3,h4]))
        th = np.squeeze(np.array([th1,th2,th3,th4]))
        C = h * np.cos(th)
        S = h*np.sin(th)
        j = np.squeeze(np.array([j1,j2,j3,j4]))

        denom = ((C[2]*S[1] - C[1]*S[2]) * (C[0]**2*S[3]-C[3]**2*S[0]) -
                (C[0]*S[3]-C[3]*S[0])*(C[2]**2*S[1] - C[1]**2*S[2]))
        a1 = 2*S[3]*(C[2]*S[1] - C[1]*S[2]) / denom
        a2 = 2*S[2]*(C[0]*S[3] - C[3]*S[0]) / denom
        a3 = -2*S[1]*(C[0]*S[3] - C[3]*S[0]) / denom
        a4 = -2*S[0]*(C[2]*S[1] - C[1]*S[2]) / denom
        a = np.array([a1,a2,a3,a4])

        if u is not None:
            d2u = (a*(u[j] - u[k])).sum()
        else:
            d2u=None

        if jacobian is True:
            # Compute FD matrix, as COO scipy sparse matrix data
            j = np.concatenate([np.array([k]),j])
            i = np.full(j.shape,k, dtype = np.intp)
            value = np.concatenate([np.array([-a.sum()]),a])
            coo = (value,i,j)
        else:
            coo=None

        return  d2u, coo

    D2 = [d2(k) for k in G.interior]

    if u is not None:
        d2u = np.array([tup[0] for tup in D2])
    else:
        d2u = None

    if jacobian is True:
        i = np.concatenate([tup[1][1] for tup in D2])
        j = np.concatenate([tup[1][2] for tup in D2])
        value = np.concatenate([tup[1][0] for tup in D2])
        M = csr_matrix((value, (i,j)), shape = [G.num_points]*2)
        M = M[M.getnnz(1)>0]
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
