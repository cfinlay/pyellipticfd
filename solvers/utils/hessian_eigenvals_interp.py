import numpy as np

__stencil = np.array([[1, 0],
                    [1, 1],
                    [0, 1],
                    [-1,1]])
__permute = np.array([[0, 1],
                    [2, 1],
                    [2, 3],
                    [0,3]])

def max(U,dx):
    """Compute the maximum eigenvalue of the function U on the interior."""
    Nx, Ny = U.shape

    #Index excluding the boundary
    I = np.arange(1,Nx-1,1,dtype=np.intp) 
    J = np.arange(1,Ny-1,1,dtype=np.intp)

    Dvv = np.zeros((Nx-2,Ny-2,__stencil.shape[0]))
    theta_tmp = np.zeros((Nx-2,Ny-2,__stencil.shape[0])) 
    A = U[1:-1,1:-1]

    for k in range(__stencil.shape[0]):

        # Neighbouring direction vectors in the stencil
        v = __stencil[__permute[k,0],:]
        w = __stencil[__permute[k,1],:]

        # Sum of antipodal points in the stencil
        B = U[np.ix_(I+v[0],J+v[1])] + U[np.ix_(I-v[0],J-v[1])]
        C = U[np.ix_(I+w[0],J+w[1])] + U[np.ix_(I-w[0],J-w[1])]

        # Get argmax parameter 0 <= t<= 1
        Case1 = np.logical_and(
                    B==C,
                    2*A+B==2*C)

        Case2 = np.logical_and(
                    np.logical_or(
                        np.logical_or(
                            2*A >=B,
                            2*A >= C),
                        B<=C),
                    np.logical_or(
                        2*A+B > 2*C,
                        B > C))

        Case3 = np.logical_and(B>C, 2*A < C)

        t = np.zeros((Nx-2,Ny-2))

        t[Case1] = .5
        t[Case2] = 1.0
        t[Case3] = (B[Case3]-C[Case3]) / (
                    -2*A[Case3] + C[Case3] + np.sqrt(
                        4*A[Case3]**2 + B[Case3]**2 - 2*(2*A[Case3]+B[Case3])*C[Case3] +
                        2*C[Case3]**2))

        # Compute the maximum second directional derivative 
        # between stencil points v & w
        Dvv[:,:,k] = (-2*A + t*B + (1-t)*C) / ((1+t**2)*dx**2)

        # Calculate the angle of the derivative
        r = t/np.sqrt(1+t**2)
        if k==0:
            theta_tmp[:,:,k] = np.arcsin(r)
        elif k==1:
            theta_tmp[:,:,k] = np.pi/2-np.arcsin(r)
        elif k==2:
            theta_tmp[:,:,k] = np.pi/2+np.arcsin(r)
        elif k==3:
            theta_tmp[:,:,k] = np.pi-np.arcsin(r)

    Kmax = Dvv.argmax(2)
    [I,J] = np.indices(Kmax.shape)
    lambda_max = Dvv[I,J,Kmax]
    theta = theta_tmp[I,J,Kmax]
    return lambda_max, theta

def min(U,dx):
    """Compute the minimum eigenvalue of the function U on the interior."""
    Nx, Ny = U.shape

    #Index excluding the boundary
    I = np.arange(1,Nx-1,1,dtype=np.intp) 
    J = np.arange(1,Ny-1,1,dtype=np.intp)

    Dvv = np.zeros((Nx-2,Ny-2,__stencil.shape[0]))
    theta_tmp = np.zeros((Nx-2,Ny-2,__stencil.shape[0])) 
    A = U[1:-1,1:-1]

    for k in range(__stencil.shape[0]):

        # Neighbouring direction vectors in the stencil
        v = __stencil[__permute[k,0],:]
        w = __stencil[__permute[k,1],:]

        # Sum of antipodal points in the stencil
        B = U[np.ix_(I+v[0],J+v[1])] + U[np.ix_(I-v[0],J-v[1])]
        C = U[np.ix_(I+w[0],J+w[1])] + U[np.ix_(I-w[0],J-w[1])]

        # Get argmin parameter 0 <= t<= 1
        Case1 = np.logical_and(
                    B==C,
                    2*A+B==2*C)

        Case2 = np.logical_or(
                    np.logical_and(
                        B < C,
                        np.logical_or(
                            2*A==B,
                            np.logical_and(
                                2*A > B,
                                2*A <= C))),
                    np.logical_and(
                        2*A+B < 2*C,
                        B > C))

        Case3 = np.logical_or(
                    np.logical_and(B==C, 2*A < C),
                    np.logical_and(B<C, 2*A < B))

        Case4 = np.logical_and(B<C, 2*A > C)

        t = np.zeros((Nx-2,Ny-2))

        t[Case1] = .5
        t[Case2] = 1.0
        t[Case3] = ((-B[Case3] + C[Case3])/(2*A[Case3]-B[Case3]) 
                    + np.abs(-2*A[Case3]+C[Case3])/
                      np.abs(B[Case3]-2*A[Case3]))
        t[Case4] = (( -2*A[Case4] + C[Case4] + np.sqrt(
                        4*A[Case4]**2 + B[Case4]**2 - 2*(2*A[Case4]+B[Case4])*C[Case4] +
                        2*C[Case4]**2)) / (C[Case4]-B[Case4]))

        # Compute the minimum second directional derivative 
        # between stencil points v & w
        Dvv[:,:,k] = (-2*A + t*B + (1-t)*C) / ((1+t**2)*dx**2)

        # Calculate the angle of the derivative
        r = t/np.sqrt(1+t**2)
        if k==0:
            theta_tmp[:,:,k] = np.arcsin(r)
        elif k==1:
            theta_tmp[:,:,k] = np.pi/2-np.arcsin(r)
        elif k==2:
            theta_tmp[:,:,k] = np.pi/2+np.arcsin(r)
        elif k==3:
            theta_tmp[:,:,k] = np.pi-np.arcsin(r)

    Kmin = Dvv.argmin(2)
    [I,J] = np.indices(Kmin.shape)
    lambda_min = Dvv[I,J,Kmin]
    theta = theta_tmp[I,J,Kmin]
    return lambda_min, theta
