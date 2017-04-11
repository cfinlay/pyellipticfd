import numpy as np
from grids import process_v
from scipy.sparse import coo_matrix

def d2(u,G,v):
    """
    Compute the second directional derivative of u, in direction v.

    Parameters
    ----------
    u : array_like
        Function values at grid points.
    G : FDMesh
        The mesh of grid points.
    v : array_like
        Direction to take second derivative.

    Returns
    -------
    d2u : array_like
        Second derivative in the direction v.
    M : scipy csr_matrix
        Finite difference matrix: the Jacobian.
    """

    # v must be an array of vectors, a direction for each interior point
    v = process_v(G,v)

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

    d2u = np.sum(weight*u_bc, axis=1)

    i = np.repeat(pairs[:,0],3)
    j = pairs.flatten()
    weight = weight.flatten()

    M = coo_matrix((weight, (i,j)), shape = [G.num_nodes]*2).tocsr()
    M = M[M.getnnz(1)>0]

    return d2u, M

def d2eigs(u,G):
    """
    Compute the maximum and minimum eigenvalues of the Hessian of U.

    Parameters
    ----------
    u : array_like
        Function values at grid points.
    G : FDMesh
        The mesh of grid points.
    eigs : string
        Specify which eigenvalue to retrieve: "min", "max", or "both".

    Returns
    -------
    Lambda : tuple
    """

    # Center point index, and stencil neighbours
    I, J = G.pairs[:,0], G.pairs[:,1:]

    X = G.vertices[J] - G.vertices[I,None] # The stencil vectors
    Xnorm = np.linalg.norm(X,axis=2)

    u_bc = u[G.pairs]
    weight = (2*np.array([-np.sum(Xnorm,axis=1),Xnorm[:,1],Xnorm[:,0]]).T /
                ( np.sum((Xnorm**2)*Xnorm[:,[1,0]],axis=1)[:,None] ) )

    d2u = np.sum(weight*u_bc, axis=1)

    def eigs(k):
        mask = I==k
        d2 = d2u[mask]
        w = weight[mask]
        ix = np.argsort(d2)
        return d2[ix[[0,-1]]], G.pairs[mask][ix[[0,-1]]], w[ix[[0,-1]]]

    data = [eigs(k) for k in G.interior]

    lambda_min = np.array([tup[0][0] for tup in data])
    lambda_max = np.array([tup[0][1] for tup in data])
    j_min = np.array([tup[1][0] for tup in data])
    j_max = np.array([tup[1][1] for tup in data])
    w_min = np.array([tup[2][0] for tup in data])
    w_max = np.array([tup[2][1] for tup in data])

    i = np.repeat(j_min[:,0],3)

    M_min = coo_matrix((w_min.flatten(), (i,j_min.flatten())), shape = [G.num_nodes]*2).tocsr()
    M_min = M_min[M_min.getnnz(1)>0]

    M_max = coo_matrix((w_max.flatten(), (i,j_max.flatten())), shape = [G.num_nodes]*2).tocsr()
    M_max = M_max[M_max.getnnz(1)>0]

    return (lambda_min, M_min), (lambda_max, M_max)

def d2min(u,G):
    """
    Compute the minimum eigenvalues of the Hessian of u.
    """
    return d2eigs(u,G)[0]

def d2max(u,G):
    """
    Compute the maximum eigenvalues of the Hessian of u.
    """
    return d2eigs(u,G)[1]
