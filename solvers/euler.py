import numpy as np
import itertools
import warnings

def euler(U,F,CFL,solution_tol=1e-4,max_iters=1e5):
    """
    Solve F[U] = 0 by iterating Euler steps until the
    stopping criteria |U^n+1 - U^n| < solution_tol.

    Parameters
    ----------
    U0 : array_like
        The initial condition.
    F : function
        An function returning the operator value on the
        interior of the domain. For example, if U is
        a m x n array, then F[U] must be (m-2) x (n-2).
    CFL : scalar
        The maximum time step determined by the CFML condition.
    solution_tol : scalar
        Stopping criterion.
    max_iters : scalar
        Maximum number of iterations.

    Returns
    -------
    U : array_like
        The solution.
    i : scalar
        Number of iterations taken.
    diff : scalar
        The maximum absolute difference between the solution
        and the previous iterate.
    """
    for i in itertools.count(1):
        U_interior = U[1:-1,1:-1] + CFL * F(U)

        diff = np.amax(np.absolute(U[1:-1,1:-1] - U_interior))
        U[1:-1,1:-1] = U_interior

        if diff < solution_tol:
            return U, i, diff
        elif i >= max_iters:
            warnings.warn("Maximum iterations reached")
            return U, i, diff
