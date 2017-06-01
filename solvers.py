import numpy as np
import itertools
import warnings
import time
from scipy.sparse.linalg import spsolve

def euler(U,F,CFL,solution_tol=1e-4,max_iters=1e5,timeout=None):
    """
    Solve F[U] = 0 by iterating Euler steps until the
    stopping criteria |U^n+1 - U^n| < solution_tol.

    Parameters
    ----------
    U0 : array_like
        The initial condition.
    F : function
        An function returning the operator value, including poins on the
        boundary.
    CFL : scalar
        The maximum time step determined by the CFL condition.
    solution_tol : scalar
        Stopping criterion, in the infinity norm.
    max_iters : scalar
        Maximum number of iterations.

    Returns
    -------
    U : array_like
        The solution.
    diff : scalar
        The maximum absolute difference between the solution
        and the previous iterate.
    i : scalar
        Number of iterations taken.
    time : scalar
        CPU time spent computing solution.
    """
    t0 = time.time()
    if not timeout is None:
        timeout = time.time()+timeout

    for i in itertools.count(1):
        U_new = U - CFL * F(U)

        diff = np.amax(np.absolute(U - U_new))
        U = U_new

        if diff < solution_tol:
            return U, diff, i, time.time()-t0
        elif i >= max_iters:
            warnings.warn("Maximum iterations reached")
            return U, diff, i, time.time()-t0
        elif (not timeout is None) and (time.time() > timeout):
            warnings.warn("Maximum computation time reached")
            return U, diff, i, time.time()-t0

def newton(U,operator,CFL,solution_tol=1e-4,max_iters=1e2, 
        euler_timeout=1, max_euler_iters=None):
    """
    Use semismooth Newton's method to find the steady state F[U]=0.

    Parameters
    ----------
    U : array_like
        The initial condition.
    operator : function
        An function returning a tuple of the operator value, and the Jacobian.
    solution_tol : scalar
        Stopping criterion, in the infinity norm.
    max_iters : scalar
        Maximum number of iterations.
    euler_timeout : scalar
        In between Newton steps, the method may perform Euler steps. 
        The scalar euler_timeout gives the ratio of the time spent on Euler over
        the time spent doing a Newton step. Defaults to 1.
    max_euler_iters : scalar
        Maximum allowable Euler iterations.

    Returns
    -------
    U : array_like
        The solution.
    diff : scalar
        The maximum absolute difference between the solution
        and the previous iterate.
    i : scalar
        Number of iterations taken.
    time : scalar
        CPU time spent computing solution.
    """
    t0 = time.time()

    if max_euler_iters is None:
        max_euler_iters=U.size

    for i in itertools.count(1):
        if euler_timeout is not None:
            tstart = time.time()

        Fu, Grad = operator(U)
        d = spsolve(Grad,-Fu)

        U_new = U + np.reshape(d,U.shape)
        diff = np.amax(np.absolute(U - U_new))
        U = U_new

        if euler_timeout is not None:
            NewtonTime = time.time()-tstart
            timeout = euler_timeout*NewtonTime
        else:
            timeout = None

        if diff < solution_tol:
            return U, diff, i, time.time()-t0
        elif i >= max_iters:
            warnings.warn("Maximum iterations reached")
            return U, diff, i, time.time()-t0

        if (max_euler_iters is not 0) or (timeout is not None):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                U, diff, _, _ = euler(U, lambda U : operator(U, jacobian=False),
                                    CFL, solution_tol, max_iters=max_euler_iters,
                                    timeout=timeout)

        if diff < solution_tol:
            return U, diff, i, time.time()-t0
        elif i+1 >= max_iters:
            warnings.warn("Maximum iterations reached")
            return U, diff, i, time.time()-t0

def policy(U,getF,getPolicy,CFL,solution_tol=1e-4,max_iters=1e5,
        euler_tol=1e-3, max_euler_iters=15):
    """
    policy(U,getF,getPolicy,CFL,solution_tol=1e-4,max_iters=1e5,
           euler_tol=1e-3, max_euler_iters=15)

    Solve F[U] = 0 by policy iteration until the
    stopping criteria |U^n+1 - U^n| < solution_tol.

    At each iteration, the policy is fixed, and the current value of U
    is iterated forward via Euler step with this frozen policy.
    The policy is then recalculated and the process begins anew.

    Parameters
    ----------
    U0 : array_like
        The initial condition.
    getF : function
        A function taking in a policy and returning an operator.
    getPolicy: function
        A function taking in values on the grid and returning a policy (for example,
        the direction of the minimum eigenvalue).
    CFL : scalar
        The maximum time step determined by the CFL condition.
    solution_tol : scalar
        Stopping criterion.
    euler_tol : scalar
        Tolerance for solving the sub-problem with fixed policy.
    max_iters : scalar
        Maximum number of iterations - the sum of Euler step iterations.
    max_euler_iters : int
        Maximum number of iterations for the Euler step.

    Returns
    -------
    U : array_like
        The solution.
    i : scalar
        Number of iterations taken.
    diff : scalar
        The maximum absolute difference between the solution
        and the previous iterate.
    policy_diff: scalar
        Maximum absolute difference between the optimal policy and the previous iterate.
    """

    Pol = getPolicy(U) # initialize policy

    iters = 0
    for i in itertools.count(0):
        F = getF(Pol)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            U, solution_diff, euler_iters,  _ = euler(U,F,CFL,
                                                   solution_tol = euler_tol,
                                                   max_iters = max_euler_iters)

        Pol_old = getPolicy(U)
        policy_diff = np.amax(np.abs(Pol - Pol_old))
        Pol = Pol_old

        iters = iters+euler_iters

        if solution_diff < solution_tol:
            break
        if iters >=max_iters:
            break

    if iters >= max_iters:
        warnings.warn("Maximum iterations reached")
    return U, iters, solution_diff, policy_diff
