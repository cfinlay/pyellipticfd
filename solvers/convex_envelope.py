import numpy as np
import directional_derivatives_interp as ddi
import directional_derivatives_grid as ddg
import warnings
from euler import *
import itertools

def euler_step(G,dx,solution_tol=1e-4,max_iters=1e5,method='grid'):
    """
    euler_step(g,dx,solution_tol=1e-4,max_iters=1e5) 
    
    Find the convex envelope of g, in 2D. The solution is calculated by
    iterating Euler steps to solve the the obstacle problem
        max(-lambda1[u],u-g) = 0
    where lambda1 is the minimum eigenvalue of the Hessian.
    
    Parameters
    ----------
    g : array_like
        A 2D array of the function g.
    dx : scalar        dx is the uniform grid spacing.
    solution_tol : scalar
        Stopping criterion.
    max_iters : int
        Maximum number of iterations.
    method : string
        Specify the monotone finite difference method of computing the minimum
        eigenvalue.  Either 'grid' or 'interpolate'.

    Returns
    -------
    u : array_like
        The convex envelope.
    iters: scalar
        Number of iterations.
    diff: scalar
        Maximum absolute difference between the solution and the previous iterate.
    """
    dt = 1/2*dx ** 2  #time step, from CFL condition
    U = G.copy()

    if method=="interpolate":
        def F(W):
            lambda_1 = ddi.d2min(W,dx)[0]
            return np.minimum(lambda_1, G[1:-1,1:-1] - W[1:-1,1:-1])
    elif method=="grid":
        def F(W):
            lambda_1 = ddg.d2min(W,dx)
            return np.minimum(lambda_1, G[1:-1,1:-1] - W[1:-1,1:-1])

    U, iters, diff = euler(U,F,dt,solution_tol=solution_tol,max_iters=max_iters)

    if iters >= max_iters:
        warnings.warn("Maximum iterations reached")
    return U, iters, diff


def policy(G,dx,solution_tol=1e-4,max_iters=1e5,policy_tol=1e-2,
           euler_tol=1e-3, max_euler_iters=15):
    """
    policy(g,dx,solution_tol=1e-4,max_iters=1e5,policy_tol=1e-2,
               euler_tol=1e-3, max_euler_iters=15)
    
    Find the convex envelope of g, in 2D. The solution is calculated by
    using policy iteration to solve the the obstacle problem
        max(-lambda1[u],u-g) = 0
    where lambda1 is the minimum eigenvalue of the Hessian.
    
    Parameters
    ----------
    g : array_like
        A 2D array of the function g.
    dx : scalar
        dx is the uniform grid spacing.
    solution_tol : scalar
        Stopping criterion for difference between succesive solutions
        of the obstacle problem.
    max_iters : int
        Maximum number of iterations.
    policy_tol : scalar
        Stopping criterion for difference between succesive policies.
    euler_tol : scalar
        Tolerance for solving the sub-problem
            (1) max(-Dvv[u], u-g)=0 
    max_euler_iterations : int
        Maximum number of iterations to solve (1) with Euler step.

    Returns
    -------
    u : array_like
        The convex envelope.
    iters: scalar
        Number of iterations.
    diff: scalar
        Maximum absolute difference between the solution and the previous iterate.
    policy_diff: scalar
        Maximum absolute difference between the optimal policy and the previous iterate.
    """
    Th = ddi.d2min(G,dx)[1] # initialize policy 
    dt = 1/2*dx ** 2  #time step, from CFL condition
    U = G.copy()
    iters = 0
    
    for i in itertools.count(1):
        def F(W):
            Dvv = ddi.d2(W,Th,dx)
            return np.minimum(Dvv, G[1:-1,1:-1] - W[1:-1,1:-1])

        U, euler_iters, solution_diff = euler(U,F,dt,
                                               solution_tol = euler_tol, max_iters=max_euler_iters)

        Th_old = ddi.d2min(U,dx)[1]
        policy_diff = np.amax(np.abs(Th - Th_old))
        Th = Th_old

        iters = iters+euler_iters

        if iters >= max_iters:
            warnings.warn("Maximum iterations reached")
            break
        elif policy_diff < policy_tol:
            break
        elif solution_diff < solution_tol:
            break

    return U, iters, solution_diff, policy_diff
