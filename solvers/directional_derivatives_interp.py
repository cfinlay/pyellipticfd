import numpy as np

# Stencil vectors
__stencil = np.array([[  1,  0],
                      [  1,  1],
                      [  0,  1],
                      [ -1,  1],
                      [ -1,  0],
                      [ -1, -1],
                      [ -1,  0],
                      [  1, -1]], dtype=np.intp)

# Arrangement of neighbouring stencil vectors: basis vector first
__nb_pts = np.array([[[1, 0], [1,1]],
                     [[0, 1], [1,1]],
                     [[0, 1], [-1,1]],
                     [[-1,0], [-1,1]],
                     [[-1,0], [-1,-1]],
                     [[0,-1], [-1,-1]],
                     [[0,-1], [1,-1]],
                     [[1,0], [1,-1]]], dtype=np.intp)

def d1da(U,dx,direction="both"):
    """
    d1da(U,dx,d,direction="both") 
    
    Calculate descent and ascent directions respectively minimizing and
    maximizing 
        <grad u, p>, st ||p|| = 1.
    
    Parameters
    ----------
    u : array_like
        Function values at grid points.
    dx : scalar
        Uniform grid resolution.
    string : string
        Specify which direction to retrieve: "descent", "ascent", or "both".

    Returns
    -------
    Value : a list, or an array
        If direction="both", a list containing the value <grad u, p>. The first
        entry of the list gives the value when p is a descent direction; the second for p
        an ascent direction.
        If direction!="both", then an array of the specified maximal or minimal value.
    Theta : a list, or an array
        If direction="both", a list containing the angle of the descent and ascent directions,
        with the angle of the descent direction first.
        If direction!="both", then an array of the angle of the corresponding direction.
    """
    Nx, Ny = U.shape
    Dv = np.zeros((16,Nx-2,Ny-2)) 
    theta = np.zeros((16,Nx-2,Nx-2))

    # Index excluding the boundary
    I = np.arange(1,Nx-1,1,dtype=np.intp) 
    J = np.arange(1,Ny-1,1,dtype=np.intp)
    
    A = U[1:-1,1:-1]
    for k, (p,nbs) in enumerate(zip(__stencil,__nb_pts)):

        # Exact directional derivative (and angle) in the stencil
        Dv[k,:,:] = (-A + U[np.ix_(I+p[0],J+p[1])])/(dx*np.sqrt(p[0]**2 + p[1]**2))
        theta[k,:,:] = np.arctan2(p[1],p[0])

        # Now check directional derivatives via linear interpolation.
        # First get neighbouring direction vectors in the stencil
        v = nbs[0,:] # +/- basis vector
        w = nbs[1,:] # diagonal

        # Function values in the stencil
        B = U[np.ix_(I+v[0],J+v[1])]
        C = U[np.ix_(I+w[0],J+w[1])]

        # Calculate angle of possible minimizer
        th = np.arctan2(B-C, A-B)
        th = np.mod(th,np.pi)
        th[th>np.pi/4] = 0
        t = np.tan(th) # convex parameter

        # Candidate for minimizer
        Dv[k+8,:,:] = (-A + t*C + (1-t)*B)/(dx*np.sqrt(1+t**2))

        # Angle of candidate minimizer
        if k==0:
            theta[k+8,:,:] = np.arctan(t)
        elif k==1:
            theta[k+8,:,:] = np.pi/2-np.arctan(t)
        elif k==2:
            theta[k+8,:,:] = np.pi/2+np.arctan(t)
        elif k==3:
            theta[k+8,:,:] = np.pi-np.arctan(t)
        elif k==4:
            theta[k+8,:,:] = np.pi+np.arctan(t)
        elif k==5:
            theta[k+8,:,:] = np.pi*3/2-np.arctan(t)
        elif k==6:
            theta[k+8,:,:] = np.pi*3/2+np.arctan(t)
        elif k==7:
            theta[k+8,:,:] = -np.arctan(t)
    theta = np.mod(theta,np.pi*2)

    [i,j] = np.indices((Nx-2,Ny-2))
    if direction=="both":
        kmin = Dv.argmin(0)
        dmin = Dv[kmin,i,j]
        theta_min = theta[kmin,i,j]

        kmax = Dv.argmax(0)
        dmax = Dv[kmax,i,j]
        theta_max = theta[kmax,i,j]

        Value = dmin, dmax
        Theta = theta_min, theta_max

        return Value, Theta
    elif direction=="descent":
        kmin = Dv.argmin(0)
        dmin = Dv[kmin,i,j]
        theta_min = theta[kmin,i,j]

        return dmin, theta_min
    elif direction=="ascent":
        kmax = Dv.argmax(0)
        dmax = Dv[kmax,i,j]
        theta_max = theta[kmax,i,j]

        return dmax, theta_max

def d1descent(U,dx):
    """
    d2descent(u,dx) 
    
    Compute the minimal value of <grad u, p> st ||p||=1.
    Equivalent to calling d2da(u,dx,direction="descent")
    """
    Value, Theta = d1da(U,dx,direction="descent")
    return Value, Theta

def d1ascent(U,dx):
    """
    d2ascent(u,dx) 
    
    Compute the maximal value of <grad u, p> st ||p||=1.
    Equivalent to calling d2da(u,dx,direction="ascent")
    """
    Value, Theta = d1da(U,dx,direction="ascent")
    return Value, Theta

def d1(U,theta,dx):
    """
    d1(u,theta,dx)

    Compute the directional derivative of u, in direction 
    v = [cos(theta), sin(theta)].

    Parameters
    ----------
    u : array_like
        Function values at grid points.
    theta : scalar
        Angle of direction of derivative.
    dx : scalar
        Uniform grid resolution.

    Returns
    -------
    du : array_like
        Directional derivative.
    """
    Nx, Ny = U.shape

    theta = np.mod(theta, 2*np.pi)

    # Parameter for adding convex combination of neighbouring stencil vectors
    th = np.mod(theta, np.pi)
    if theta.size==1:
        if th <=np.pi/4 or th >=3*np.pi/4:
            t = np.abs(np.tan(th))
        else:
            t = np.abs(np.cos(th)/np.sin(th))
    else:
        t = np.zeros(th.shape)
        ix = np.logical_or(th<=np.pi/4, th>=3*np.pi/4)
        nix = np.logical_not(ix)
        t[ix] = np.abs(np.tan(th[ix]))
        t[nix] = np.abs(np.cos(th[nix])/np.sin(th[nix]))

    # Directional derivative along theta
    DU = np.zeros((Nx-2,Ny-2)) 
    
    # Index excluding the boundary
    I = np.arange(1,Nx-1,1,dtype=np.intp) 
    J = np.arange(1,Ny-1,1,dtype=np.intp)

    # Determine where theta falls in the stencil
    Cases = (np.logical_and(theta >=0, theta < np.pi/4),
             np.logical_and(theta >=np.pi/4, theta < np.pi/2),
             np.logical_and(theta >=np.pi/2, theta < 3*np.pi/4),
             np.logical_and(theta >=3*np.pi/4, theta < np.pi),
             np.logical_and(theta >=np.pi, theta < 5*np.pi/4),
             np.logical_and(theta >=5*np.pi/4, theta < 3*np.pi/2),
             np.logical_and(theta >=3*np.pi/2, theta < 7*np.pi/4),
             np.logical_and(theta >=7*np.pi/4, theta < 2*np.pi))

    A = U[1:-1,1:-1]
    for ix, nbs in zip(Cases, __nb_pts):
        if (ix.size != 1) or ix:
            # Neighbouring direction vectors in the stencil
            v = nbs[0,:] # basis vector
            w = nbs[1,:] # either (1,1) or (-1,1)

            # Function values
            B = U[np.ix_(I+v[0],J+v[1])]
            C = U[np.ix_(I+w[0],J+w[1])]

            if (t.size==1):
                DU = (-A + (1-t)*B + t*C)/(np.sqrt((1+t**2))*dx)
            else:
                DU[ix] = (-2*A[ix] + (1-t[ix])*B[ix] +t[ix]*C[ix])/(np.sqrt((1+t[ix]**2))*dx)

    return DU

def d2(U,theta,dx):
    """
    d2(u,theta,dx)

    Compute the second directional derivative of u, in direction 
    v = [cos(theta), sin(theta)].

    Parameters
    ----------
    u : array_like
        Function values at grid points.
    theta : scalar
        Angle of direction of derivative.
    dx : scalar
        Uniform grid resolution.

    Returns
    -------
    d2u : array_like
        Second directional derivative.
    """
    Nx, Ny = U.shape

    theta = np.mod(theta,np.pi)

    # Parameter for adding convex combination of neighbouring stencil vectors
    if theta.size==1:
        if theta <=np.pi/4 or theta >=3*np.pi/4:
            t = np.abs(np.tan(theta))
        else:
            t = np.abs(np.cos(theta)/np.sin(theta))
    else:
        t = np.zeros(theta.shape)
        ix = np.logical_or(theta<=np.pi/4, theta>=3*np.pi/4)
        nix = np.logical_not(ix)
        t[ix] = np.abs(np.tan(theta[ix]))
        t[nix] = np.abs(np.cos(theta[nix])/np.sin(theta[nix]))

    # Second directional derivative along theta
    D2U = np.zeros((Nx-2,Ny-2)) 
    
    # Index excluding the boundary
    I = np.arange(1,Nx-1,1,dtype=np.intp) 
    J = np.arange(1,Ny-1,1,dtype=np.intp)

    # Determine where theta falls in the stencil
    Cases = (np.logical_and(theta >=0, theta < np.pi/4),
             np.logical_and(theta >=np.pi/4, theta < np.pi/2),
             np.logical_and(theta >=np.pi/2, theta < 3*np.pi/4),
             np.logical_and(theta >=3*np.pi/4, theta < np.pi))

    A = U[1:-1,1:-1]
    for ix, nbs in zip(Cases, __nb_pts[0:4,:,:]):
        if (ix.size != 1) or ix:
            # Neighbouring direction vectors in the stencil
            v = nbs[0,:] # basis vector
            w = nbs[1,:] # either (1,1) or (-1,1)

            # Sum of antipodal points in the stencil
            B = U[np.ix_(I+v[0],J+v[1])] + U[np.ix_(I-v[0],J-v[1])]
            C = U[np.ix_(I+w[0],J+w[1])] + U[np.ix_(I-w[0],J-w[1])]

            if (t.size==1):
                D2U = (-2*A + (1-t)*B + t*C)/((1+t**2)*dx**2)
            else:
                D2U[ix] = (-2*A[ix] + (1-t[ix])*B[ix] +t[ix]*C[ix])/((1+t[ix]**2)*dx**2)

    return D2U

def d2eigs(U,dx,eigs="both"):
    """
    d2eigs(U,dx,eigs="both") 
    
    Compute the maximum and minimum eigenvalues of the Hessian of U.
    
    Parameters
    ----------
    u : array_like
        Function values at grid points.
    dx : scalar
        Uniform grid resolution.
    eigs : string
        Specify which eigenvalue to retrieve: "min", "max", or "both".

    Returns
    -------
    Lambda : a list, or an array
        If eigs="both", a list containing the minimal and maximal eigenvalues,
        with the minimal eigenvalue first.
        If eigs!="both", then an array of the specified eigenvalue.
    Theta : a list, or an array
        If eigs="both", a list containing the angle of the minimal and maximal eigenvalues,
        with the angle of the minimal eigenvalue first.
        If eigs!="both", then an array of the angle of the specified eigenvalue.
    """
    Nx, Ny = U.shape
    Dvv = np.zeros((8,Nx-2,Ny-2))
    theta = np.zeros((8,Nx-2,Ny-2)); 

    #Index excluding the boundary
    I = np.arange(1,Nx-1,1,dtype=np.intp) 
    J = np.arange(1,Ny-1,1,dtype=np.intp)

    A = U[1:-1,1:-1]

    for k, (p,nbs) in enumerate(zip(__stencil[0:4,:],__nb_pts[0:4,:])):

        # Exact directional derivative (and angle) in the stencil
        Dvv[k,:,:] = ((-2*A + U[np.ix_(I+p[0],J+p[1])] + U[np.ix_(I-p[0],J-p[1])]) 
                        / ((p[0]**2 + p[1]**2)*dx**2))
        theta[k,:,:] = np.arctan2(p[1],p[0])

        # Now check directional derivatives via linear interpolation.
        # First get neighbouring direction vectors in the stencil
        v = nbs[0,:] # basis vector
        w = nbs[1,:] # either (1,1) or (-1,1)

        # Sum of antipodal function values in the stencil
        B = U[np.ix_(I+v[0],J+v[1])] + U[np.ix_(I-v[0],J-v[1])]
        C = U[np.ix_(I+w[0],J+w[1])] + U[np.ix_(I-w[0],J-w[1])]

        # Calculate angle of possible minimizer
        th = 1/2 * np.arctan2(B-C, 2*A-B)
        th = np.mod(th,np.pi/2)
        th[th>np.pi/4] = 0
        t = np.tan(th) # convex parameter

        # Candidate for minimizer
        Dvv[k+4,:,:] = (-2*A + t*C + (1-t)*B)/((1+t**2)* dx**2)

        # Angle of candidate minimizer
        if k==0:
            theta[k+4,:,:] = np.arctan(t)
        elif k==1:
            theta[k+4,:,:] = np.pi/2-np.arctan(t)
        elif k==2:
            theta[k+4,:,:] = np.pi/2+np.arctan(t)
        elif k==3:
            theta[k+4,:,:] = np.mod(np.pi-np.arctan(t),np.pi)

    [i,j] = np.indices((Nx-2,Ny-2))

    if eigs=="both":
        kmin = Dvv.argmin(0)
        lambda_min = Dvv[kmin,i,j]
        theta_min = theta[kmin,i,j]

        kmax = Dvv.argmax(0)
        lambda_max = Dvv[kmax,i,j]
        theta_max = theta[kmax,i,j]

        Lambda = lambda_min, lambda_max
        Theta = theta_min, theta_max

        return Lambda, Theta
    elif eigs=="min":
        kmin = Dvv.argmin(0)
        lambda_min = Dvv[kmin,i,j]
        theta_min = theta[kmin,i,j]

        return lambda_min, theta_min
    elif eigs=="max":
        kmax = Dvv.argmax(0)
        lambda_max = Dvv[kmax,i,j]
        theta_max = theta[kmax,i,j]

        return lambda_max, theta_max

def d2min(U,dx):
    """
    d2min(u,dx) 
    
    Compute the minimum eigenvalues of the Hessian of U.
    Equivalent to calling d2eigs(u,dx,eigs="min")
    """
    return d2eigs(U,dx,eigs="min")

def d2max(U,dx):
    """
    d2max(u,dx) 
    
    Compute the maximum eigenvalues of the Hessian of U.
    Equivalent to calling d2eigs(u,dx,eigs="max")
    """
    return d2eigs(U,dx,eigs="max")
