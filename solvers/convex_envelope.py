import numpy as np
import scipy.sparse as sparse
import itertools
import warnings

import directional_derivatives_interp as ddi
import directional_derivatives_grid as ddg
import finite_difference_matrices as fdm
from euler import euler
from policy import policy
from newton import newton
from linesolver import convex_linesolver

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
            Fw = W - G
            Fw[1:-1,1:-1] = np.maximum(-lambda_1, Fw[1:-1,1:-1])
            return Fw
    elif method=="grid":
        def F(W):
            lambda_1 = ddg.d2min(W,dx)[0]
            Fw = W - G
            Fw[1:-1,1:-1] = np.maximum(-lambda_1, Fw[1:-1,1:-1])
            return Fw

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
    if method=="interpolate":
        def getF(policy):
            def F(W):
                Wvv = ddi.d2(W,policy,dx)
                Fw = W - G
                Fw[1:-1,1:-1]= np.maximum(-Wvv, Fw[1:-1,1:-1])
                return Fw
            return F
    elif method=="grid":
        def getF(policy):
            def F(W):
                Wvv = ddg.d2(W,ddg.stencil,dx,ix=policy)
                Fw = W - G
                Fw[1:-1,1:-1]= np.maximum(-Wvv, Fw[1:-1,1:-1])
                return Fw
            return F

    if method=="interpolate":
        def getPolicy(U):
            return ddi.d2min(U,dx)[1]
    elif method=="grid":
        def getPolicy(U):
            return ddg.d2min(U,dx)[1]

    dt = 1/2*dx ** 2  #time step, from CFL condition
    U = G.copy()
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
    I,J = np.indices(G.shape)
    ix_int = np.ravel_multi_index((I[1:-1,1:-1],J[1:-1,1:-1]),G.shape)
    ix_int = np.reshape(ix_int,ix_int.size)

    def operator(U,getGrad=True):
        lambda1, Ix = ddg.d2min(U,dx)

        M = fdm.d2(G.shape, ddg.stencil, dx, Ix)

        Fu = U-G
        Fu_int = Fu[1:-1,1:-1]
        b = -lambda1 > Fu_int
        Fu_int[b] = -lambda1[b]

        if getGrad:
            Fu = Fu.reshape(Fu.size)
            b = np.reshape(b,b.size)

            Grad = sparse.eye(G.size,format='csr')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") #suppress stupid warning
                Grad[ix_int[b],:] = -M[b,:]

            return Fu, Grad
        else:
            return Fu

    U = G.copy()
    return newton(U,operator,1/2*dx**2,**kwargs)

def line_solver(G,stencil,solution_tol=1e-6, max_iters=1e3):
    U = np.copy(G)
    for i in itertools.count(1):
        Uold = U
        U = convex_linesolver(U,stencil)
        err = np.max(np.abs(U-Uold))

        if err < solution_tol:
            return U, i, err
        elif i >= max_iters:
            warnings.warn("Maximum iterations reached")
            return U, i, err
