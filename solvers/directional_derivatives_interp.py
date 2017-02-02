import numpy as np

# Stencil vectors -- only half the stencil needed
__stencil = np.array([[1, 0],
                    [1, 1],
                    [0, 1],
                    [-1,1]])

# Arrangement of neighbouring stencil vectors
__permute = np.array([[0, 1],
                    [2, 1],
                    [2, 3],
                    [0,3]])

## Fix this -- convex parameter t is incorrect
#def d2(U,theta,dx):
#    Nx, Ny = U.shape
#
#    theta = np.mod(theta,np.pi)
#
#    # Parameter for adding convex combination of neighbouring stencil vectors
#    t = np.abs(np.tan(theta))
#
#    # Second directional derivative along theta
#    D2U = np.zeros((Nx-2,Ny-2)) 
#    
#    # Index excluding the boundary
#    I = np.arange(1,Nx-1,1,dtype=np.intp) 
#    J = np.arange(1,Ny-1,1,dtype=np.intp)
#
#    # Determine where theta falls in the stencil
#    Cases = (np.logical_and(theta >=0, theta < np.pi/4),
#             np.logical_and(theta >=np.pi/4, theta < np.pi/2),
#             np.logical_and(theta >=np.pi/2, theta < 3*np.pi/4),
#             np.logical_and(theta >=3*np.pi/4, theta < np.pi))
#
#    A = U[1:-1,1:-1]
#    for ix, p in zip(Cases,__permute):
#        if (ix.size != 1) or ix:
#            # Neighbouring direction vectors in the stencil
#            v = __stencil[p[0],:]
#            w = __stencil[p[1],:]
#
#            # Sum of antipodal points in the stencil
#            B = U[np.ix_(I+v[0],J+v[1])] + U[np.ix_(I-v[0],J-v[1])]
#            C = U[np.ix_(I+w[0],J+w[1])] + U[np.ix_(I-w[0],J-w[1])]
#
#        if (t.size==1):
#            D2U = (-2*A + t*B + (1-t)*C)/((1+t)*dx)**2
#        elif t.size==D2U.size:
#            D2U[ix] = (-2*A[ix] + t[ix]*B[ix] +(1-t[ix])*C[ix])/((1+t[ix])*dx)**2
#
#    return D2U

def d2eigs(U,dx):
    """Compute the maximum and minimum eigenvalues of the Hessian of U, on the interior."""
    Nx, Ny = U.shape
    Dvv = np.zeros((8,Nx-2,Ny-2))
    theta = np.zeros((8,Nx-2,Ny-2)); 

    #Index excluding the boundary
    I = np.arange(1,Nx-1,1,dtype=np.intp) 
    J = np.arange(1,Ny-1,1,dtype=np.intp)

    A = U[1:-1,1:-1]

    for k, (p,perm) in enumerate(zip(__stencil,__permute)):

        # Exact directional derivative (and angle) in the stencil
        Dvv[k,:,:] = ((-2*A + U[np.ix_(I+p[0],J+p[1])] + U[np.ix_(I-p[0],J-p[1])]) 
                        / ((p[0]**2 + p[1]**2)*dx**2))
        theta[k,:,:] = np.arctan2(p[0],p[1])

        # Now check directional derivatives via linear interpolation.
        # First get neighbouring direction vectors in the stencil
        v = __stencil[perm[0],:] # basis vector
        w = __stencil[perm[1],:] # either (1,1) or (-1,1)

        # Sum of antipodal function values in the stencil
        B = U[np.ix_(I+v[0],J+v[1])] + U[np.ix_(I-v[0],J-v[1])]
        C = U[np.ix_(I+w[0],J+w[1])] + U[np.ix_(I-w[0],J-w[1])]

        # Calculate angle of possible minimizer
        th = 1/2 * np.arctan2(C-B, 2*A-C)
        th = np.mod(th,np.pi/2)
        th[th>np.pi/4] = 0
        t = np.tan(th) # convex parameter

        # Candidate for minimizer
        Dvv[k+4,:,:] = (-2*A + t*B + (1-t)*C)/((1+t**2)* dx**2)

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

    kmin = Dvv.argmin(0)
    lambda_min = Dvv[kmin,i,j]
    theta_min = theta[kmin,i,j]

    kmax = Dvv.argmax(0)
    lambda_max = Dvv[kmax,i,j]
    theta_max = theta[kmax,i,j]

    Lambda = lambda_min, lambda_max
    Theta = theta_min, theta_max
    return Lambda, Theta

def d2min(U,dx):
    Lambda, Theta = d2eigs(U,dx)
    return Lambda[0], Theta[0]

def d2max(U,dx):
    Lambda, Theta = d2eigs(U,dx)
    return Lambda[1], Theta[1]
