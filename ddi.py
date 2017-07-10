"""Functions to calculate finite differences with linear interpolation."""

import numpy as np
from scipy.sparse import csr_matrix

from pyellipticfd import _ddutils

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
        "boundary", or "all". If not specified, defaults to "interior".

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
    else:
        Ix = np.arange(G.num_points)

    # Get finite difference simplices on correct domain
    if not (domain=="boundary" or domain=="interior"):
        I, S = G.simplices[:,0], G.simplices[:,1:]
    else:
        mask = np.in1d(G.simplices[:,0], Ix)
        simplices = G.simplices[mask]
        I, S = simplices[:,0], simplices[:,1:]

    X = G.points[S] - G.points[I,None] # The simplex vectors
    X = np.swapaxes(X,1,2)                 # Transpose last two axis

    if (domain=="interior" or domain=="boundary"):
        # dictionary, to look up domain index from graph index
        d = dict(zip(Ix,range(Ix.size)))
        i = [d[key] for key in I]

        # Cone coordinates of direction to take derivative
        Xi = np.linalg.solve(X,v[i])
    else:
        Xi = np.linalg.solve(X,v[I])

    # Given interior index, compute directional derivative
    def d1(k):
        mask = I==k
        xi = Xi[mask] # cone coordinates
        x = X[mask]   # stencil vectors
        s = simplices[mask] # simplex indices

        mask_f = (xi>=0).all(axis=1) # Farkas' lemma

        # If the centre point is not on the boundary,
        # choose the first available simplex.
        if np.in1d(k,G.interior).all():
            xi_f =  xi[mask_f][0] # Neighbour weights
            i_f = s[mask_f][0][1:] # index of neighbours
        else:
            # Otherwise we need to be a bit more careful.
            in_boundary = np.in1d(s[mask_f,1:],G.bdry)
            mask_boundary = in_boundary.reshape((sum(mask_f),G.dim))
            mask_interior = np.logical_not(mask_boundary.all(axis=1))

            i_f = s[mask_f]
            i_f = np.squeeze(i_f[mask_interior,1:])

            xi_f = xi[mask_f]
            xi_f = np.squeeze(xi_f[mask_interior])

        h = 1/np.sum(xi_f)  # distance to interpolation point

        if u is not None:
            u_f = u[i_f].dot(xi_f)
            d1u = u[k]/h-u_f
        else:
            d1u = None

        if jacobian is True:
            # Compute FD matrix, as COO scipy sparse matrix data
            j = np.concatenate([np.array([k]),i_f])
            i = np.full(j.shape,k, dtype = np.intp)
            value = -np.concatenate([np.array([-1])/h, xi_f])
            coo = (value,i,j)
        else:
            coo=None


        return  d1u, coo

    D1 = [d1(k) for k in Ix]

    if u is not None:
        d1u = np.array([tup[0] for tup in D1])
    else:
        d1u = None

    if jacobian is True:
        i = np.concatenate([tup[1][1] for tup in D1])
        j = np.concatenate([tup[1][2] for tup in D1])
        value = np.concatenate([tup[1][0] for tup in D1])
        M = csr_matrix((value, (i,j)), shape = [G.num_points]*2)
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

def d1grad(G,u,jacobian=False,control=False):
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

    # number of directions to take, per stencil
    Nds = np.ceil(1/G.angular_resolution) # effectively dtheta^2
    if Nds < 1:
        Nds = 1

    xi = np.linspace(0,1,Nds+1) # stencil coordinates of directions
    xi = np.array([1-xi,xi])
    Nds +=1

    # Index of centre point and simplex neighbours
    I, S = G.simplices[:,0], G.simplices[:,1:]

    X = G.points[S] - G.points[I,None] # The simplex vectors
    X = np.swapaxes(X,1,2)                 # Transpose last two axis

    V = np.einsum('ijk,kl->ijl',X,xi)
    V = V/np.linalg.norm(V,axis=1)[:,None,:]

    def grad(k):
        mask = I==k
        V_ = np.concatenate(V[mask],axis=1)
        X_ = X[mask]
        S_ = S[mask]
        Ns = X_.shape[0] # number of simplices about the point
        bcst_shape = np.concatenate([[Ns],V_.shape])

        jf = np.repeat(range(Ns),Nds)
        Xi_f = np.linalg.solve(X_[jf],V_.T)  # simplex coordinates
        S_f = S_[jf,:]
        h_f = 1./Xi_f.sum(axis=1)

        u_f = np.einsum('ij,ij->i',u[S_f],Xi_f)
        d1u = ( u_f - u[k]/h_f)

        isort = d1u.argsort()
        imax = isort[-1]

        d1max = d1u[imax]

        if jacobian is True:
            # Compute FD matrix, as COO scipy sparse matrix data
            j = np.concatenate([np.array([k]),S_f[imax]])
            i = np.full(j.shape,k, dtype = np.intp)
            val = np.concatenate([np.array([-1/h_f[imax] ]), Xi_f[imax] ])
            coo = (val,i,j)
        else:
            coo = None

        if control is True:
            v = V_[:,imax]
        else:
            v = None

        return d1max,coo, v

    e = [grad(k) for k in G.interior]

    d1_max = np.array([tup[0] for tup in e])

    if jacobian is True:
        i = np.concatenate([tup[1][1] for tup in e])
        j = np.concatenate([tup[1][2] for tup in e])
        val = np.concatenate([tup[1][0] for tup in e])
        M = csr_matrix((val, (i,j)), shape = [G.num_points]*2)
        M = M[M.getnnz(1)>0]
    else:
        M = None

    if control is True:
        v = np.array([tup[2] for tup in e])
    else:
        v = None

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
    X = np.swapaxes(X,1,2)                 # Transpose last two axis

    # dictionary, to look up interior index from graph index
    d = dict(zip(G.interior,range(G.num_interior)))
    i = [d[key] for key in I]

    # Cone coordinates of direction to take derivative
    Xi = np.linalg.solve(X,v[i])

    # Given interior index, compute directional derivative
    def d2(k):
        mask = I==k
        xi = Xi[mask] # cone coordinates
        x = X[mask]   # stencil vectors
        s = interior_simplices[mask] # simplex indices

        # Forward direction
        mask_f = np.squeeze((xi>=0).all(axis=1)) # Farkas' lemma
        xi_f =  xi[mask_f][0]
        h_f = 1/np.sum(xi_f)

        # Backward direction
        mask_b = np.squeeze((xi<=0).all(axis=1))
        xi_b =  xi[mask_b][0]
        h_b = -1/np.sum(xi_b)

        i_f = s[mask_f][0][1:]
        i_b = s[mask_b][0][1:]

        if u is not None:
            u_f = u[i_f].dot(xi_f)
            u_b = u[i_b].dot(-xi_b)

            d2u = 2/(h_b+h_f)*(u_f + u_b - u[k]*(1/h_f + 1/h_b) )
        else:
            d2u=None

        if jacobian is True:
            # Compute FD matrix, as COO scipy sparse matrix data
            j = np.concatenate([np.array([k]),i_f, i_b])
            i = np.full(j.shape,k, dtype = np.intp)
            value = np.concatenate([np.array([-(1/h_f + 1/h_b)]), xi_f, -xi_b])*2/(h_f+h_b)
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


def d2eigs(G,u,jacobian=False, control=False):
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

    Returns
    -------
    lambda_min, lambda_max
        Respectively the minimal and maximal eigenvalues.
        These are tuples consisting of the value, Jacobian and control.
    """

    if G.dim==3:
        raise NotImplementedError("Eigenvalues not yet implemented in 3d.")

    # number of directions to take, per stencil
    Nds = np.ceil(1/G.angular_resolution) # effectively dtheta^2
    if Nds < 1:
        Nds = 1

    xi = np.linspace(0,1,Nds+1) # stencil coordinates of directions
    xi = np.array([1-xi,xi])
    Nds +=1

    # Index of centre point and simplex neighbours
    I, S = G.simplices[:,0], G.simplices[:,1:]

    X = G.points[S] - G.points[I,None] # The simplex vectors
    X = np.swapaxes(X,1,2)                 # Transpose last two axis

    V = np.einsum('ijk,kl->ijl',X,xi)
    V = V/np.linalg.norm(V,axis=1)[:,None,:]

    def eigs(k):
        mask = I==k
        V_ = np.concatenate(V[mask],axis=1)
        X_ = X[mask]
        S_ = S[mask]
        Ns = X_.shape[0] # number of simplices about the point
        bcst_shape = np.concatenate([[Ns],V_.shape])

        jf = np.repeat(range(Ns),Nds)
        Xi_f = np.linalg.solve(X_[jf],V_.T)  # simplex coordinates
        S_f = S_[jf,:]
        h_f = 1./Xi_f.sum(axis=1)

        V_bc = np.broadcast_to(V_,bcst_shape)
        Xi_ = np.linalg.solve(X_,-V_bc)  # simplex coordinates
        ixb = np.where((Xi_>=0).all(axis=1))
        _, j = np.unique(ixb[1], return_index=True)
        jb = ixb[0][j]
        Xi_b = Xi_[jb,:,np.arange(Nds*Ns,dtype=np.intp)]
        S_b = S_[jb,:]
        h_b = 1./Xi_b.sum(axis=1)

        u_f = np.einsum('ij,ij->i',u[S_f],Xi_f)
        u_b = np.einsum('ij,ij->i',u[S_b],Xi_b)
        d2u = 2/(h_b+h_f)*( u_f + u_b - u[k]*(1/h_f +1/h_b))

        isort = d2u.argsort()
        imin, imax = isort[[0,-1]]

        d2min, d2max = d2u[[imin,imax]]

        if jacobian is True:
            # Compute FD matrix, as COO scipy sparse matrix data
            j_min = np.concatenate([np.array([k]),S_f[imin], S_b[imin]])
            i_min = np.full(j_min.shape,k, dtype = np.intp)

            h_fm = h_f[imin]
            h_bm = h_b[imin]
            xi_fm = Xi_f[imin]
            xi_bm = Xi_b[imin]

            val_min = 2/(h_fm+h_bm)*np.concatenate([np.array([-(1/h_fm + 1/h_bm)]),
                                                    xi_fm, xi_bm])
            coo_min = (val_min,i_min,j_min)


            j_max = np.concatenate([np.array([k]),S_f[imax], S_b[imax]])
            i_max = np.full(j_max.shape,k, dtype = np.intp)

            h_fp = h_f[imax]
            h_bp = h_b[imax]
            xi_fp = Xi_f[imax]
            xi_bp = Xi_b[imax]

            val_max = 2/(h_fp+h_bp)*np.concatenate([np.array([-(1/h_fp + 1/h_bp)]),
                                                    xi_fp, xi_bp])
            coo_max = (val_max,i_max,j_max)
        else:
            coo_min, coo_max = [None]*2

        if control is True:
            v_min = V_[:,imin]
            v_max = V_[:,imax]
        else:
            v_min, v_max = [None]*2

        return (d2min,coo_min, v_min), (d2max,coo_max, v_max)

    e = [eigs(k) for k in G.interior]

    d2min = np.array([tup[0][0] for tup in e])
    d2max = np.array([tup[1][0] for tup in e])

    if jacobian is True:
        i_min = np.concatenate([tup[0][1][1] for tup in e])
        j_min = np.concatenate([tup[0][1][2] for tup in e])
        val_min = np.concatenate([tup[0][1][0] for tup in e])
        M_min = csr_matrix((val_min, (i_min,j_min)), shape = [G.num_points]*2)
        M_min = M_min[M_min.getnnz(1)>0]

        i_max = np.concatenate([tup[1][1][1] for tup in e])
        j_max = np.concatenate([tup[1][1][2] for tup in e])
        val_max = np.concatenate([tup[1][1][0] for tup in e])
        M_max = csr_matrix((val_max, (i_max,j_max)), shape = [G.num_points]*2)
        M_max = M_max[M_max.getnnz(1)>0]
    else:
        M_min, M_max = [None]*2

    if control is True:
        v_min = np.array([tup[0][2] for tup in e])
        v_max = np.array([tup[1][2] for tup in e])
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
