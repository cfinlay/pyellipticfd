def euler_step(G,dx,tol=1e-6,max_iters=1e4):
    """
    euler_step(g,dx,tol=1e-6,max_iters=1e4) 
    
    Find the convex envelope of g, in 2D. The solution is calculated by time iterating
    the obstacle problem
        max(-lambda1[u],u-g) = 0
    where lambda1 is the minimum eigenvalue of the Hessian.
    
    Parameters
    ----------
    g : array_like
        A 2D array of the function g.
    dx : scalar
        dx is the uniform grid spacing.
    tol : scalar
        Stopping criterion.
    max_iters : int
        Maximum number of iterations.

    Returns
    -------
    u : array_like
        The convex envelope.
    """
    import numpy as np

    Nx = G.shape[0]
    Ny = G.shape[1]

    dt = dx ** 2  #time step, from CFL condition

    #Index excluding the boundary
    I_int = np.arange(1,Nx-1,1,dtype=np.intp) 
    J_int = np.arange(1,Ny-1,1,dtype=np.intp)

    # Define the stencil -- one sided only
    stencil = np.array([[0,1],
                        [1,1],
                        [1,0],
                        [-1,1]])

    # Preallocate memory
    Uold = np.copy(G)
    Dvv = np.zeros((Nx-2,Ny-2,stencil.shape[0]))

    # Now iterate until a steady state is reached
    iters = 0
    while (iters < max_iters):
        for k in range(stencil.shape[0]):
            v = stencil[k,:]
            norm_v = np.linalg.norm(v)
            
            Uc = Uold[np.ix_(I_int,J_int)]
            Uf = Uold[np.ix_(I_int+v[0],J_int+v[1])]
            Ub = Uold[np.ix_(I_int-v[0],J_int-v[1])]        

            Dvv[:,:,k] = (- 2 * Uc + Ub + Uf) / ((norm_v * dx) ** 2)
        
        lambda_1 = np.amin(Dvv, axis=2)

        Uint = Uold[1:-1,1:-1] + dt * np.minimum(lambda_1,G[1:-1,1:-1] - Uold[1:-1,1:-1])
        diff = np.amax(np.absolute(Uold[1:-1,1:-1] - Uint))

        if diff < tol:
            break
       
        Uold[1:-1,1:-1] = Uint
        iters = iters + 1

    U = np.copy(G)
    U[1:-1,1:-1] = Uint

    return U
