import numpy as np
import scipy.sparse as sparse

import directional_derivatives_interp as ddi
import directional_derivatives_grid as ddg
import finite_difference_matrices as fdm
from euler import euler
from policy import policy
from newton import newton

def euler_step(G,dx,method='grid',**kwargs):
    """
    Find the convex envelope of g, in 2D. The solution is calculated by
    iterating Euler steps to solve the the obstacle problem
        max(-lambda1[u],u-g) = 0
    where lambda1 is the minimum eigenvalue of the Hessian.

    Parameters
    ----------
    g : array_like
        A 2D array of the function g.
    dx : scalar
        dx is the uniform grid spacing.
    method : string
        Specify the monotone finite difference method of computing the minimum
        eigenvalue.  Either 'grid' or 'interpolate'.
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
    dt = 1/2*dx ** 2  #time step, from CFL condition
    U = G.copy()

    if method=="interpolate":
        def F(W):
            lambda_1 = ddi.d2min(W,dx)[0]
            return np.minimum(lambda_1, G[1:-1,1:-1] - W[1:-1,1:-1])
    elif method=="grid":
        def F(W):
            lambda_1 = ddg.d2min(W,dx)[0]
            return np.minimum(lambda_1, G[1:-1,1:-1] - W[1:-1,1:-1])

    return euler(U,F,dt,**kwargs)


def policy_iteration(G,dx,method='grid',**kwargs):
    """
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
    method : string
        Specify the monotone finite difference method.  Either 'grid' or 'interpolate'.
    solution_tol : scalar
        Stopping criterion for difference between succesive iterates.
    max_iters : int
        Maximum number of iterations - the sum of Euler step iterations.
    euler_tol : scalar
        Tolerance for solving the sub-problem
            (1) max(-Dvv[u], u-g)=0
    max_euler_iters : int
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
    dt = 1/2*dx ** 2  #time step, from CFL condition
    U = G.copy()

    if method=="interpolate":
        def getF(policy):
            def F(W):
                Uvv = ddi.d2(W,policy,dx)
                return np.minimum(Uvv, G[1:-1,1:-1] - W[1:-1,1:-1])
            return F
    elif method=="grid":
        def getF(policy):
            def F(W):
                Uvv = ddg.d2(W,ddg.stencil,dx,policy)
                return np.minimum(Uvv, G[1:-1,1:-1] - W[1:-1,1:-1])
            return F

    if method=="interpolate":
        def getPolicy(U):
            return ddi.d2min(U,dx)[1]
    elif method=="grid":
        def getPolicy(U):
            return ddg.d2min(U,dx)[1]

    return policy(U,getF,getPolicy,dt,**kwargs)

def newton_method(G,dx,**kwargs):
    """
    Find the convex envelope of g, in 2D. The solution is calculated by
    semismooth Newton's method to solve the the obstacle problem
        max(-lambda1[u],u-g) = 0
    where lambda1 is the minimum eigenvalue of the Hessian.

    Parameters
    ----------
    G : array_like
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
    shape = G.shape
    Nx = shape[0]
    Ny = shape[1]
    N = np.prod(shape)


    def operator(U):
        lambda1, Ix = ddg.d2min(U,dx)

        M = fdm.d2(U.shape, ddg.stencil, dx, Ix)

        b = -lambda1 > U[1:-1,1:-1]-G[1:-1,1:-1]
        Fu = U[1:-1,1:-1]-G[1:-1,1:-1]
        Fu[b] = -lambda1[b]

        b = np.reshape(b,(Nx-2)*(Ny-2))
        Fu = np.reshape(Fu,(Nx-2)*(Ny-2))

        ix_int = fdm.domain_indices(shape,1)[0]
        Grad = sparse.eye(N,format='csr')
        Grad = Grad[ix_int,:]
        Grad[b,:] = -M[b,:]
        Grad = Grad[:,ix_int]

        return Fu, Grad

    U = G.copy()
    return newton(U,operator,**kwargs)
