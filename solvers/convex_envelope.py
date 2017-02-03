import numpy as np
import directional_derivatives_interp as ddi
import warnings

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
    Nx, Ny = G.shape

    dt = dx ** 2  #time step, from CFL condition

    # Preallocate memory
    Uold = np.copy(G)

    # Now iterate until a steady state is reached
    iters = 0
    while (iters < max_iters):
        lambda_1 = ddi.d2min(Uold,dx)[0]

        Uint = Uold[1:-1,1:-1] + dt * np.minimum(lambda_1,G[1:-1,1:-1] - Uold[1:-1,1:-1])
        diff = np.amax(np.absolute(Uold[1:-1,1:-1] - Uint))

        if diff < tol:
            break
       
        Uold[1:-1,1:-1] = Uint
        iters = iters + 1

    U = np.copy(G)
    U[1:-1,1:-1] = Uint

    if iters >= max_iters:
        warnings.warn("Maximum iterations reached without attaining specified tolerance")
    return U
