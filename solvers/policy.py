import numpy as np
from euler import euler
import itertools
import warnings

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
            U, euler_iters, solution_diff = euler(U,F,CFL,
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
