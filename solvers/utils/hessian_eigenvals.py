import numpy as np

stencil = np.array([[1, 0],
                    [1, 1],
                    [0, 1],
                    [-1,1]])
permute = np.array([[0, 1],
                    [2, 1],
                    [2, 3],
                    [0,3]])

def max(U,dx):
    """Compute the maximum eigenvalue of the function U on the interior."""
    Nx, Ny = U.shape

    #Index excluding the boundary
    I = np.arange(1,Nx-1,1,dtype=np.intp) 
    J = np.arange(1,Ny-1,1,dtype=np.intp)

    Dvv = np.zeros((Nx-2,Ny-2,stencil.shape[0]))
    A = U[1:-1,1:-1]

    for k in range(stencil.shape[0]):

        # Neighbouring direction vectors in the stencil
        v = stencil[permute[k,0],:]
        w = stencil[permute[k,1],:]

        # Sum of antipodal points in the stencil
        B = U[np.ix_(I+v[0],J+v[1])] + U[np.ix_(I-v[0],J-v[1])]
        C = U[np.ix_(I+w[0],J+w[1])] + U[np.ix_(I-w[0],J-w[1])]

        Case1 = np.logical_or( 
                    np.logical_and(
                        (B > C),
                        np.logical_or(
                            np.logical_and(
                                2*A >= C,
                                2*A < B),
                            2*A > B)),
                    np.logical_and(
                        2*A + B >= 2*C,
                        B <= C))

        Case2 = np.logical_or(
                    np.logical_and(
                        2*A <= C,
                        B < C),
                    np.logical_and(
                        B <= C,
                        2*A+B < 2*C))

        Case3 = np.logical_and(
                    B > C,
                    2*A < C)

        Dvv_tmp = np.zeros((Nx-2,Ny-2))

        Dvv_tmp[Case1] = 1/(2*dx ** 2) * (-2*A[Case1] + B[Case1])
        Dvv_tmp[Case2] = 1/(dx ** 2) * (-2*A[Case2] + C[Case2])
        Dvv_tmp[Case3] = 1/(2*dx ** 2) * ( -2*A[Case3] + C[Case3] +
                                        np.sqrt(4*A[Case3]**2 + B[Case3]**2 - 
                                            4*A[Case3]*C[Case3] - 
                                            2*B[Case3]*C[Case3] + 2*C[Case3]**2))
        Dvv[:,:,k] = Dvv_tmp

    lambda_max = np.amax(Dvv, axis = 2)
    return lambda_max

def min(U,dx):
    """Compute the minimum eigenvalue of the function U on the interior."""
    Nx, Ny = U.shape

    #Index excluding the boundary
    I = np.arange(1,Nx-1,1,dtype=np.intp) 
    J = np.arange(1,Ny-1,1,dtype=np.intp)

    Dvv = np.zeros((Nx-2,Ny-2,stencil.shape[0]))
    A = U[1:-1,1:-1]

    for k in range(stencil.shape[0]):

        # Neighbouring direction vectors in the stencil
        v = stencil[permute[k,0],:]
        w = stencil[permute[k,1],:]

        # Sum of antipodal points in the stencil
        B = U[np.ix_(I+v[0],J+v[1])] + U[np.ix_(I-v[0],J-v[1])]
        C = U[np.ix_(I+w[0],J+w[1])] + U[np.ix_(I-w[0],J-w[1])]

        Case1 = np.logical_or(
                    np.logical_or( 
                        np.logical_and(
                            2*A <= B,
                            B < C),
                        np.logical_and(
                            B <= C,
                            2*A <= C)),
                    np.logical_and(
                        B < C,
                        2*A + B <= 2*C))

        Case2 = np.logical_and(
                    B > C,
                    np.logical_or(
                        np.logical_and(
                            2*A+B > 2*C,
                            2*A < C),
                        2*A > C))

        Case3 = np.logical_and(
                    B <= C,
                    2*A > C)

        Dvv_tmp = np.zeros((Nx-2,Ny-2))

        Dvv_tmp[Case1] = 1/(2*dx ** 2) * (-2*A[Case1] + B[Case1])
        Dvv_tmp[Case2] = 1/(dx ** 2) * (-2*A[Case2] + C[Case2])
        Dvv_tmp[Case3] = 1/(2*dx ** 2) * ( -2*A[Case3] + C[Case3] -
                                        np.sqrt(4*A[Case3]**2 + B[Case3]**2 - 
                                            4*A[Case3]*C[Case3] - 
                                            2*B[Case3]*C[Case3] + 2*C[Case3]**2))
        Dvv[:,:,k] = Dvv_tmp

    lambda_min = np.amax(Dvv, axis = 2)
    return lambda_min
