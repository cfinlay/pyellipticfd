import numpy as np
import directional_derivatives_interp as ddi
import warnings
from euler import *
import itertools

def euler_step(G,dx,solution_tol=1e-4,max_iters=1e5):
    """
    euler_step(g,dx,solution_tol=1e-4,max_iters=1e5) 
    
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
    solution_tol : scalar
        Stopping criterion.
    max_iters : int
        Maximum number of iterations.

    Returns
    -------
    u : array_like
        The convex envelope.
    iters: scalar
        Number of iterations.
    diff: scalar
        Maximum absolute difference between the solution and the previous iterate.
    """
    dt = dx ** 2  #time step, from CFL condition

    def F(W):
        lambda_1 = ddi.d2min(W,dx)[0]
        return np.minimum(lambda_1, G[1:-1,1:-1] - W[1:-1,1:-1])

    U, iters, diff = euler(G,F,dt,solution_tol=solution_tol,max_iters=max_iters)
    return U, iters, diff

def policy(G,dx,solution_tol=1e-4,max_iters=1e5,policy_tol=1e-2,
           euler_tol=1e-3, max_euler_iters=1e3):

    Th = ddi.d2min(G,dx)[1] # initialize policy 
    dt = dx ** 2  #time step, from CFL condition
    U = G
    iters = 0
    
    for i in itertools.count(1):
        def F(W):
            Dvv = ddi.d2(W,Th,dx)
            return np.minimum(Dvv, G[1:-1,1:-1] - W[1:-1,1:-1])

        U_old, euler_iters, euler_diff = euler(U,F,dt,
                                               solution_tol = dx**2, max_iters=max_euler_iters)
        solution_diff = np.amax(np.abs(U_old - U))
        U = U_old

        Th_old = ddi.d2min(U,dx)[1]
        policy_diff = np.amax(np.abs(Th - Th_old))
        Th = Th_old

        iters = iters+euler_iters

        if iters > max_iters:
            warnings.warn("Maximum iterations reached")
            break
        elif solution_diff < solution_tol:
            break
        elif policy_diff < policy_tol:
            break

    return U, iters, solution_diff, policy_diff
