"""Functions to calculate finite differences on regular grids."""

import numpy as np
from scipy.sparse import coo_matrix

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
        is returned.
    jacobian : boolean
        Whether to also return the Jacobian M.
    domain : string
        Which nodes to compute derivative on: one of "interior",
        "boundary", or "all". If not specified, defaults to "interior".

    Returns
    -------
    d1u : array_like
        First derivative in the direction v.
    M : scipy csr_matrix
        Finite difference matrix. Only returned if jacobian is True, or if u is
        not specified.
    """
    if u is None:
        jacobian = True

    # v must be an array of vectors, a direction for each point
    v = _ddutils.process_v(G,v,domain=domain)

    if domain=="interior":
        Ix = G.interior
    elif domain=="boundary":
        Ix = G.boundary
    else:
        Ix = np.arange(G.num_nodes)

    # Get finite difference simplices on interior
    if not (domain=="boundary" or domain=="interior"):
        I, J = G.neighbours[:,0], G.neighbours[:,1]
    else:
        mask = np.in1d(G.neighbours[:,0], Ix)
        neighbours = G.neighbours[mask]
        I, J = neighbours[:,0], neighbours[:,1]

    X = G.vertices[J] - G.vertices[I] # The simplex vectors
    Xs = X/np.linalg.norm(X,axis=1)[:,None]

    if (domain=="interior" or domain=="boundary"):
        # dictionary, to look up domain index from graph index
        d = dict(zip(Ix,range(Ix.size)))
        i = [d[key] for key in I]

        # Cosine of direction against stencil vectors
        C = np.einsum('ij,ij->i',Xs,v[i])
    else:
        C = np.einsum('ij,ij,->i',Xs,v[I])

    # Given interior index, compute directional derivative
    def d1(k):
        mask = I==k
        c = C[mask] # cosine, masked
        x = X[mask] # stencil vectors
        nbs = neighbours[mask] # neighbour indices

        ix = np.argmax(c)
        x = x[ix]
        nbs = nbs[ix]

        h = np.linalg.norm(x)

        if u is not None:
            d1u = u[nbs].dot([1, -1])/h
        else:
            d1u = None

        if jacobian is True:
            # Compute FD matrix, as COO scipy sparse matrix data
            i = np.full(2, k, dtype = np.intp)
            value = np.array([1, -1])/h
            coo = (value,i,nbs)
        else:
            coo=None

        return  d1u, coo

    D1 = [d1(k) for k in Ix]

    if u is not None:
        d1u = np.array([tup[0] for tup in D1])

    if jacobian is True:
        i = np.concatenate([tup[1][1] for tup in D1])
        j = np.concatenate([tup[1][2] for tup in D1])
        value = np.concatenate([tup[1][0] for tup in D1])
        M = coo_matrix((value, (i,j)), shape = [G.num_nodes]*2).tocsr()
        M = M[M.getnnz(1)>0]

    if (u is not None) and (jacobian is True):
        return d1u, M
    elif jacobian is False:
        return d1u
    else:
        return M

def d1n(G,u=None,**kwargs):
    """
    Compute the directional derivative with respect to the outward normal direction
    of the boundary of the domain. If no function is given, then the Jacobian
    is returned.

    Parameters
    ----------
    G : FDPointCloud
        The mesh of grid points.
    u : array_like
        Function values at grid points. If not specified, only the Jacobian
        is returned.
    jacobian : boolean
        Whether to also return the Jacobian M.

    Returns
    -------
    d1u : array_like
        First derivative in the direction v.
    M : scipy csr_matrix
        Finite difference matrix. Only returned if jacobian is True, or if u is
        not specified.
    """
    return d1(G, -G.boundary_normals, u=u, domain='boundary', **kwargs)

def d1grad(G,u=None,jacobian=False,control=False):
    """
    Compute the first derivative of U in the direction of the gradient.

    Parameters
    ----------
    G : FDPointCloud
        The mesh of grid points.
    u : array_like
        Function values at grid points.
    jacobian : boolean
        If True, also return the Jacobian (finite difference matrix).
    control : boolean
        If True, also return the gradient direction.

    Returns
    -------
    d1grad : array_like
        The value of the first derivative of U in the gradient direction.
    M : scipy csr_matrix
        Finite difference matrix. Only returned if jacobian is True.
    v : array_like
        The gradient direction. Only returned if control is True.
    """
    # Center point index, and stencil neighbours
    I, J = G.neighbours[:,0], G.neighbours[:,1]

    X = G.vertices[J] - G.vertices[I] # The stencil vectors
    Xnorm = np.linalg.norm(X,axis=1)

    if control is True:
        V = X/Xnorm[:,None]

    u_bc = u[G.neighbours]
    dx = 1/Xnorm
    weight = np.array([-dx,dx]).T

    d1u = np.sum(weight*u_bc, axis=1)

    def grad(k):
        mask = I==k
        d1 = d1u[mask]

        ix = np.argsort(d1)
        ix = ix[-1] # index of the largest directional derivative

        d1 = d1[ix]

        if jacobian is True:
            neighbours = G.neighbours[mask][ix]
            w = weight[mask][ix]
        else:
            pairs_, w_ = [None]*2

        if control is True:
            v = V[mask][ix]
        else:
            v = None

        return (d1, neighbours, w, v)


    data = [grad(k) for k in G.interior]

    d1_max = np.array([tup[0] for tup in data])

    if jacobian is True:
        j = np.array([tup[1] for tup in data])
        w = np.array([tup[2] for tup in data])

        i = np.repeat(j[:,0],2)

        M = coo_matrix((w.flatten(), (i,j.flatten())), shape = [G.num_nodes]*2).tocsr()
        M = M[M.getnnz(1)>0]

    if control is True:
        v = np.array([tup[3] for tup in data])

    if control is True and jacobian is True:
        return d1_max, M, v
    elif control is True and jacobian is False:
        return d1_max, v
    elif control is False and jacobian is True:
        return d1_max, M
    else:
        return d1_max

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
        is returned.
    jacobian : boolean
        Whether to also return the Jacobian M.

    Returns
    -------
    d2u : array_like
        Second derivative value in the direction v. Only returned if u is specified.
    M : scipy csr_matrix
        Finite difference matrix. Only returned if jacobian is True, or if u is
        not specified.
    """
    if u is None:
        jacobian = True

    # v must be an array of vectors, a direction for each interior point
    v = _ddutils.process_v(G,v)

    # Center point index, and stencil neighbours
    I, J = G.pairs[:,0], G.pairs[:,1:]

    X = G.vertices[J] - G.vertices[I,None] # The stencil vectors
    Xnorm = np.linalg.norm(X,axis=2)
    Xs = X/Xnorm[:,:,None] # Normalized, for computing cosine later

    # dictionary, to look up interior index from graph index
    d = dict(zip(G.interior,range(G.num_interior)))
    i = [d[key] for key in I]

    # Cone coordinates of direction to take derivative
    v_broadcast = v[i]

    cos_sq = np.einsum('lji,li->lj',Xs,v_broadcast)**2
    sum_cos2 = np.sum(cos_sq,axis=1)


    # Given interior index, get the best direction
    def indices(k):
        mask = I==k
        i = np.argmax(sum_cos2[mask])
        cfb = G.pairs[mask][i] # index of center, forward and backward points
        xnorm = Xnorm[mask][i]
        return cfb, xnorm

    data = [indices(k) for k in G.interior]

    pairs = np.array([tup[0] for tup in data])
    norms = np.array([tup[1] for tup in data])

    u_bc = u[pairs]
    weight = (2*np.array([-np.sum(norms,axis=1),norms[:,1],norms[:,0]]).T /
                ( np.sum((norms**2)*norms[:,[1,0]],axis=1)[:,None] ) )

    if u is not None:
        d2u = np.sum(weight*u_bc, axis=1)

    if jacobian is True:
        i = np.repeat(pairs[:,0],3)
        j = pairs.flatten()
        weight = weight.flatten()

        M = coo_matrix((weight, (i,j)), shape = [G.num_nodes]*2).tocsr()
        M = M[M.getnnz(1)>0]

    if (u is not None) and (jacobian is True):
        return d2u, M
    elif jacobian is False:
        return d2u
    else:
        return M

def d2eigs(G,u,jacobian=False,control=False):
    """
    Compute the eigenvalues of the Hessian of U.

    Parameters
    ----------
    G : FDPointCloud
        The mesh of grid points.
    u : array_like
        Function values at grid points.
    jacobian : boolean
        Whether to return the Jacobian (finite difference matrix) for each eigenvalue.
    control : boolean
        Whether to return the directions of the eigenvalues.

    Returns
    -------
    lambda_min, lambda_max
        Respectively the minimal and maximal eigenvalues.
        These are tuples if either jacobian or control is true.
    """

    # Center point index, and stencil neighbours
    I, J = G.pairs[:,0], G.pairs[:,1:]

    X = G.vertices[J] - G.vertices[I,None] # The stencil vectors
    Xnorm = np.linalg.norm(X,axis=2)

    if control is True:
        V = X[:,0,:]/Xnorm[:,0,None]

    u_bc = u[G.pairs]
    weight = (2*np.array([-np.sum(Xnorm,axis=1),Xnorm[:,1],Xnorm[:,0]]).T /
                ( np.sum((Xnorm**2)*Xnorm[:,[1,0]],axis=1)[:,None] ) )

    d2u = np.sum(weight*u_bc, axis=1)

    def eigs(k):
        mask = I==k
        d2 = d2u[mask]

        ix = np.argsort(d2)
        ix = ix[[0,-1]] # indices of the smallest and largest eigenvalue directons

        d2 = d2[ix]

        if jacobian is True:
            pairs = G.pairs[mask][ix]
            w = weight[mask][ix]
        else:
            pairs_, w_ = [None]*2

        if control is True:
            v = V[mask][ix]
        else:
            v = None

        return (d2, pairs, w, v)


    data = [eigs(k) for k in G.interior]

    lambda_min = np.array([tup[0][0] for tup in data])
    lambda_max = np.array([tup[0][1] for tup in data])

    if jacobian is True:
        j_min = np.array([tup[1][0] for tup in data])
        j_max = np.array([tup[1][1] for tup in data])
        w_min = np.array([tup[2][0] for tup in data])
        w_max = np.array([tup[2][1] for tup in data])

        i = np.repeat(j_min[:,0],3)

        M_min = coo_matrix((w_min.flatten(), (i,j_min.flatten())), shape = [G.num_nodes]*2).tocsr()
        M_min = M_min[M_min.getnnz(1)>0]

        M_max = coo_matrix((w_max.flatten(), (i,j_max.flatten())), shape = [G.num_nodes]*2).tocsr()
        M_max = M_max[M_max.getnnz(1)>0]

    if control is True:
        v_min = np.array([tup[3][0] for tup in data])
        v_max = np.array([tup[3][1] for tup in data])

    if control is True and jacobian is True:
        return (lambda_min, M_min, v_min), (lambda_max, M_max, v_max)
    elif control is True and jacobian is False:
        return (lambda_min, v_min), (lambda_max, v_max)
    elif control is False and jacobian is True:
        return (lambda_min, M_min), (lambda_max, M_max)
    else:
        return lambda_min, lambda_max

def d2min(G,u,**kwargs):
    """
    Compute the minimum eigenvalues of the Hessian of u.
    """
    return d2eigs(u,G,**kwargs)[0]

def d2max(G,u,**kwargs):
    """
    Compute the maximum eigenvalues of the Hessian of u.
    """
    return d2eigs(u,G,**kwargs)[1]
