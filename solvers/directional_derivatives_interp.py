import numpy as np
from scipy.sparse import coo_matrix

#def d1da(U,dx,direction="both"):
#    """
#    Calculate descent and ascent directions respectively minimizing and
#    maximizing
#        <grad u, p>, st ||p|| = 1.
#    """
#
#def d1descent(U,dx):
#    """
#    Compute the minimal value of <grad u, p> st ||p||=1.
#    Equivalent to calling d2da(u,dx,direction="descent")
#    """
#
#def d1ascent(U,dx):
#    """
#    Compute the maximal value of <grad u, p> st ||p||=1.
#    Equivalent to calling d2da(u,dx,direction="ascent")
#    """
#
#def d1(U,theta,dx):
#    """
#    Compute the directional derivative of u, in direction
#    v = [cos(theta), sin(theta)].
#    """

def d2(u,G,v):
    """
    Compute the second directional derivative of u, in direction v.

    Parameters
    ----------
    U : array_like
        Function values at grid points.
    G : FDMesh
        The mesh of grid points.
    v : array_like
        Direction to take second derivative.

    Returns
    -------
    d2u : array_like
        Second directional derivative.
    M : scipy csr_matrix
        Finite difference matrix.
    """


    # v must be an array of vectors, a direction for each interior point
    v = np.array(v)
    if (v.size==1) & (G.dim==2):
        # v is a constant spherical coordinate, convert to vector for each point
        v = np.broadcast_to([np.cos(v), np.sin(v)], (G.num_interior, G.dim))

    elif (v.size==2) & (G.dim==3):
        v = np.broadcast_to([np.sin(v[0])*np.cos(v[1]), np.sin(v[0])*np.sin(v[1]), np.cos(v[1])],
                     (G.num_interior, G.dim))

    elif v.size==G.dim:
        # v is already a vector, but constant for each point.
        # Broadcast to everypoint
        norm = np.linalg.norm(v)
        v = v/norm
        v = np.broadcast_to(v, (G.num_interior, G.dim))

    elif (v.size==G.num_interior) & (G.dim==2):
        # v is in spherical coordinates, convert to vector
        v = np.array([np.cos(v),np.sin(v)]).T

    elif (v.shape==(G.num_interior,2)) & (G.dim==3):
        v = np.array([np.sin(v[:,0])*np.cos(v[:,1]),
                      np.sin(v[:,0])*np.sin(v[:,1]),
                      np.cos(v[:,1])]).T

    elif v.shape==(G.num_interior,G.dim):
        #then v is a vector for each point, normalize
        norm = np.linalg.norm(v,axis=1)
        v = v/norm[:,None]



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

        # finite difference distance
        h = np.min([h_f, h_b])

        # Convex coordinates
        xi_f = np.append(1-np.sum(h*xi_f),h*xi_f)
        xi_b = np.append(1-np.sum(-h*xi_b),-h*xi_b)

        i_f = s[mask_f][0]
        i_b = s[mask_b][0]

        u_f = u[i_f].dot(xi_f)
        u_b = u[i_b].dot(xi_b)

        d2u = (-2*u[k]+u_f+u_b)/h**2

        # Compute FD matrix, as
        # COO scipy sparse matrix data
        j = np.concatenate([np.array([k]),i_f, i_b])
        i = np.full(j.shape,k, dtype = np.intp)
        val = np.concatenate([np.array([-2]), xi_f, xi_b])/h**2
        coo = (val,i,j)

        return  d2u, coo

    D2 = [d2(k) for k in G.interior]

    d2u = np.array([tup[0] for tup in D2])

    i = np.concatenate([tup[1][1] for tup in D2])
    j = np.concatenate([tup[1][2] for tup in D2])
    val = np.concatenate([tup[1][0] for tup in D2])

    M = coo_matrix((val, (i,j)), shape = [G.num_nodes]*2).tocsr()
    M = M[M.getnnz(1)>0]

    return d2u, M




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
