"""Functions to calculate interpolating finite differences."""

import numpy as np
from scipy.sparse import coo_matrix

from ddutils import process_v


def d1(u,G,v, jacobian=True, domain="interior"):
    """
    Compute the directional derivative of u, in direction v.

    Parameters
    ----------
    u : array_like
        Function values at grid points.
    G : FDPointCloud
        The mesh of grid points.
    v : array_like
        Direction to take second derivative.
    jacobian : boolean
        Switch, whether to compute the Jacobian
    domain : string
        Which nodes to compute derivative on: one of "interior",
        "boundary", or "all". If not specified, defaults to "interior".

    Returns
    -------
    d1u : array_like
        First derivative in the direction v.
    M : scipy csr_matrix
        Finite difference matrix. Only returned if jacobian==True
    """

    # v must be an array of vectors, a direction for each point
    v = process_v(G,v,domain=domain)

    if domain=="interior":
        Ix = G.interior
    elif domain=="boundary":
        Ix = G.boundary
    else:
        Ix = np.arange(G.num_nodes)

    # Get finite difference simplices on interior
    if not (domain=="boundary" or domain=="interior"):
        I, S = G.simplices[:,0], G.simplices[:,1:]
    else:
        mask = np.in1d(G.simplices[:,0], Ix)
        simplices = G.simplices[mask]
        I, S = simplices[:,0], simplices[:,1:]

    X = G.vertices[S] - G.vertices[I,None] # The simplex vectors
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

        # Forward direction
        mask_f = np.squeeze((xi>=0).all(axis=1)) # Farkas' lemma
        xi_f =  xi[mask_f][0]
        h = 1/np.sum(xi_f)

        i_f = s[mask_f][0][1:]

        u_f = u[i_f].dot(xi_f)

        d1u = -u[k]/h+u_f

        if jacobian==True:
            # Compute FD matrix, as COO scipy sparse matrix data
            j = np.concatenate([np.array([k]),i_f])
            i = np.full(j.shape,k, dtype = np.intp)
            val = np.concatenate([np.array([-1])/h, xi_f])
            coo = (val,i,j)
        else:
            coo=None

        return  d1u, coo

    D1 = [d1(k) for k in Ix]

    d1u = np.array([tup[0] for tup in D1])

    if jacobian==True:
        i = np.concatenate([tup[1][1] for tup in D1])
        j = np.concatenate([tup[1][2] for tup in D1])
        val = np.concatenate([tup[1][0] for tup in D1])
        M = coo_matrix((val, (i,j)), shape = [G.num_nodes]*2).tocsr()
        M = M[M.getnnz(1)>0]

        return d1u, M
    else:
        return d1u

def d2(u,G,v,jacobian=True):
    """
    Compute the second directional derivative of u, in direction v.

    Parameters
    ----------
    u : array_like
        Function values at grid points.
    G : FDPointCloud
        The mesh of grid points.
    v : array_like
        Direction to take second derivative.
    jacobian : boolean
        Switch, to compute the Jacobian

    Returns
    -------
    d2u : array_like
        Second derivative in the direction v.
    M : scipy csr_matrix
        Finite difference matrix. Only returned if jacobian==True
    """
    # v must be an array of vectors, a direction for each interior point
    v = process_v(G,v)

    # Get finite difference simplices on interior
    mask = np.in1d(G.simplices[:,0], G.interior)
    interior_simplices = G.simplices[mask]
    I, S = interior_simplices[:,0], interior_simplices[:,1:]

    X = G.vertices[S] - G.vertices[I,None] # The simplex vectors
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

        u_f = u[i_f].dot(h_f*xi_f)
        u_b = u[i_b].dot(-h_b*xi_b)

        d2u = 2/(h_b+h_f)*((u_f-u[k])/h_f +(u_b-u[k])/h_b)

        if jacobian==True:
            # Compute FD matrix, as COO scipy sparse matrix data
            j = np.concatenate([np.array([k]),i_f, i_b])
            i = np.full(j.shape,k, dtype = np.intp)
            val = np.concatenate([np.array([-(1/h_f + 1/h_b)]), xi_f, -xi_b])*2/(h_f+h_b)
            coo = (val,i,j)
        else:
            coo=None

        return  d2u, coo

    D2 = [d2(k) for k in G.interior]

    d2u = np.array([tup[0] for tup in D2])

    if jacobian==True:
        i = np.concatenate([tup[1][1] for tup in D2])
        j = np.concatenate([tup[1][2] for tup in D2])
        val = np.concatenate([tup[1][0] for tup in D2])
        M = coo_matrix((val, (i,j)), shape = [G.num_nodes]*2).tocsr()
        M = M[M.getnnz(1)>0]

        return d2u, M
    else:
        return d2u




## TODO: -deal with points near boundary
#def d2eigs(U,dx,stencil=stencil,eigs="both"):
#    """
#    Compute the maximum and minimum eigenvalues of the Hessian of U.
#
#    Parameters
#    ----------
#    u : array_like
#        Function values at grid points.
#    dx : scalar
#        Uniform grid resolution.
#    eigs : string
#        Specify which eigenvalue to retrieve: "min", "max", or "both".
#
#    Returns
#    -------
#    Lambda : a tuple, or an array
#        If eigs="both", a tuple containing the minimal and maximal eigenvalues,
#        with the minimal eigenvalue first.
#        If eigs!="both", then an array of the specified eigenvalue.
#    Control : a list of controls
#        If eigs="both", a tuple containing the controls of the minimal and maximal eigenvalues,
#        minimal eigenvalue first.
#        If eigs!="both", then the control of the specified eigenvalue.
#    """
#        return lambda_max, (kmax, tmax)
#
#def d2min(U,dx,**kwargs):
#    """
#    Compute the minimum eigenvalues of the Hessian of U.
#    Equivalent to calling d2eigs(u,dx,eigs="min")
#    """
#    return d2eigs(U,dx,**kwargs,eigs="min")
#
#def d2max(U,dx,**kwargs):
#    """
#    Compute the maximum eigenvalues of the Hessian of U.
#    Equivalent to calling d2eigs(u,dx,eigs="max")
#    """
#    return d2eigs(U,dx,**kwargs,eigs="max")
