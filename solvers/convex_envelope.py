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
    from utils import diff as df
    Nx = G.shape[0]
    Ny = G.shape[1]

    dt = dx ** 2 / 2  #time step, from CFL condition

    #Index excluding the boundary
    Int = np.ix_(np.arange(1,Nx-1,dtype = np.intp),
                 np.arange(1,Ny-1,dtype = np.intp)) #interior index
    # Define the stencil -- one sided only

    stencil = np.array([[0,1],
                    [1,1],
                    [1,0],
                    [-1,1],
                    [1,2],
                    [2,1],
                    [2,-1],
                    [-1,2]])
    # Preallocate memory
    Uold = np.copy(G) #Initial condition (obstacle)
    Dvv = 10**10 * np.ones((len(stencil),Nx,Ny))  #Initialize Dvv matrix for each stencil vector. Initialize with large values to make minimum choose only values where derivative is computed.
    # Now iterate until a steady state is reached
    iters = 0
    while (iters < max_iters):
        #Compute all of the directional derivatives in each subdomain
        for k in range(len(stencil)):             
            v = stencil[k,:] #grid vector
            norm_v = np.linalg.norm(v)   #length of the vector
            w = np.linalg.norm(v,np.inf) #width of the vector
            
            Ix,Iy = np.ix_(np.arange(w,Nx-w,dtype = np.intp),
                           np.arange(w,Ny-w,dtype = np.intp)) #index where v is valid
            Dvv[k,Ix,Iy] = df.Uvv(Uold,v)/((norm_v*dx)**2)
            
        lambda_1 = np.amin(Dvv,axis=0)
        Uint = Uold[Int] + dt * np.minimum(lambda_1[Int],G[Int] - Uold[Int])
        diff = np.max(np.absolute(Uold[Int] - Uint))
        if iters % (np.floor(max_iters/10)) == 0:
            print(str(iters) + '/' + str(max_iters) + '. Diff = ' + str(diff) +'.')
        if diff < tol:
            break
       
        Uold[Int] = Uint #update previous solution
        iters += 1
        
    U = Uold

    print('Stopped at ' + str(iters) + ' iterations. The residual was ' +str(diff) +'.')
    return U
