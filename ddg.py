"""Functions to calculate finite differences on regular grids."""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

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

    # Get finite difference stencils on interior
    mask = np.in1d(G.neighbours[:,0], Ix)
    neighbours = G.neighbours[mask]
    I, J = neighbours[:,0], neighbours[:,1]

    X = G.points[J] - G.points[I] # The stencil vectors
    h = np.linalg.norm(X,axis=1)
    Xs = X/h[:,None]

    # dictionary, to look up domain index from graph index
    d = dict(zip(Ix,range(Ix.size)))
    i = [d[key] for key in I]

    # Cosine of direction against stencil vectors
    C = np.einsum('ij,ij->i',Xs,v[i])
    ind = np.lexsort((-C,I))

    I = I[ind]
    J = J[ind]
    h = h[ind]

    m = np.concatenate([[True],I[1:]>I[0:-1]])
    I, J, h = I[m], J[m], h[m]

    w = 1/h
    if u is not None:
        d1u = w*(u[J] - u[I])
    else:
        d1u = None

    if jacobian:
        i = np.tile(I,2)
        j = np.concatenate([J,I])
        val = np.concatenate([w,-w])
        M = csr_matrix((val, (i,j)), shape = [G.num_points]*2)
        M = M[M.getnnz(1)>0]
    else:
        M = None

    return d1u, M

def d1n(G,u=None,**kwargs):
    """
    Compute the directional derivative with respect to the outward normal direction
    of the boundary of the domain. If no function is given, then only the Jacobian
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
        Finite difference matrix. Only calculate if jacobian is True, or if u is
        not specified.
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
        Finite difference matrix. Only calculated if jacobian is True.
    v : array_like
        The gradient direction. Only calculated if control is True.
    """
    Ix = G.interior

    # Get finite difference stencils on interior
    mask = np.in1d(G.neighbours[:,0], Ix)
    neighbours = G.neighbours[mask]
    I, J = neighbours[:,0], neighbours[:,1]

    X = G.points[J] - G.points[I] # The stencil vectors
    h = np.linalg.norm(X,axis=1)

    w = 1/h
    d1u = w*(u[J] - u[I])
    ind = np.lexsort((-d1u,I))

    I = I[ind]

    m = np.concatenate([[True],I[1:]>I[:-1]])
    d1_max = d1u[ind][m]

    if control or jacobian:
        w = w[ind][m]

    if control:
        v = w[:,None]*X[ind][m]
    else:
        v = None

    if jacobian:
        I = I[m]
        J = J[ind][m]
        i = np.tile(I,2)
        j = np.concatenate([J,I])
        val = np.concatenate([w,-w])
        M = csr_matrix((val, (i,j)), shape = [G.num_points]*2)
        M = M[M.getnnz(1)>0]
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
    if G.pairs is None:
        raise TypeError('The grid was has no finite difference pairs. Construct the grid with "interpolation=False"')

    if u is None:
        jacobian = True

    # v must be an array of vectors, a direction for each interior point
    v = _ddutils.process_v(G,v)

    # Center point index, and stencil neighbours
    I, J = G.pairs[:,0], G.pairs[:,1:]

    X = G.points[J] - G.points[I,None] # The stencil vectors
    h = np.linalg.norm(X,axis=2)
    Xs = X/h[:,:,None] # Normalized, for computing cosine later

    # dictionary, to look up interior index from graph index
    d = dict(zip(G.interior,range(G.num_interior)))
    i = [d[key] for key in I]

    # Cone coordinates of direction to take derivative
    v = v[i]

    cos_sq = np.einsum('lji,li->lj',Xs,v)**2
    sum_cos2 = np.sum(cos_sq,axis=1)

    ind = np.lexsort((-sum_cos2,I))

    I = I[ind]

    m = np.concatenate([[True],I[1:]>I[0:-1]])
    I, J, h = I[m], J[ind][m], h[ind][m]

    w = 1/h
    w0 = 2/h.sum(1)
    if u is not None:
        d2u = w0 * np.sum((u[J] - u[I,None])*w,axis=1)
    else:
        d2u = None

    if jacobian:
        i = np.tile(np.repeat(I,2),2)
        j = np.concatenate([J.flatten(),np.repeat(I,2)])
        val = np.concatenate([w.flatten(),-w.flatten()])
        M = coo_matrix((val,(i,j)), shape = (G.num_interior,G.num_points)).tocsr()
        M = diags(w0,format="csr").dot(M)
    else:
        M = None

    return d2u, M

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
        Whether to calculate the Jacobian (finite difference matrix) for each eigenvalue.
    control : boolean
        Whether to calculate the directions of the eigenvalues.

    Returns
    -------
    lambda_min, lambda_max
        Respectively the minimal and maximal eigenvalues.
        These are tuples consisting of the value, Jacobian and control.
    """

    # Center point index, and stencil neighbours
    I, J = G.pairs[:,0], G.pairs[:,1:]

    X = G.points[J] - G.points[I,None] # The stencil vectors
    h = np.linalg.norm(X,axis=2)

    w = 1/h
    w0 = 2/h.sum(1)
    d2u = w0 * np.sum((u[J] - u[I,None])*w,axis=1)

    ind = np.lexsort((-d2u,I))

    I = I[ind]

    m = I[:-1]<I[1:]
    mmin = np.concatenate([m,[True]])
    mmax = np.concatenate([[True],m])

    d2u = d2u[ind]
    lambda_min, lambda_max = d2u[mmin], d2u[mmax]

    if control or jacobian:
        w = w[ind]

    if control:
        X = X[ind]
        what = w[mmin,0]
        v_min, v_max = X[mmin,0]*what[:,None], X[mmax,0]*what[:,None]
    else:
        v_min, v_max = [None]*2

    if jacobian:
        w0 = w0[ind]
        J = J[ind]

        wmin, wmax = w[mmin], w[mmax]
        w0min, w0max = w0[mmin], w0[mmax]
        Imin, Imax = I[mmin], I[mmax]
        Jmin, Jmax = J[mmin], J[mmax]

        i = np.tile(np.repeat(Imin,2),2)
        j = np.concatenate([Jmin.flatten(),np.repeat(Imin,2)])
        val = np.concatenate([wmin.flatten(),-wmin.flatten()])
        M = coo_matrix((val,(i,j)), shape = (G.num_interior,G.num_points)).tocsr()
        M_min = diags(w0min,format="csr").dot(M)

        i = np.tile(np.repeat(Imax,2),2)
        j = np.concatenate([Jmax.flatten(),np.repeat(Imax,2)])
        val = np.concatenate([wmax.flatten(),-wmax.flatten()])
        M = coo_matrix((val,(i,j)), shape = (G.num_interior,G.num_points)).tocsr()
        M_max = diags(w0max,format="csr").dot(M)
    else:
        M_min, M_max = [None]*2

    return (lambda_min, M_min, v_min), (lambda_max, M_max, v_max)

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
