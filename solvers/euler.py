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
    i : scalar
        Number of iterations taken.
    diff : scalar
        The maximum absolute difference between the solution
        and the previous iterate.
    """
    for i in itertools.count(1):
        U_new = U - CFL * F(U)

        diff = np.amax(np.absolute(U - U_new))
        U = U_new

        if diff < solution_tol:
            return U, i, diff
        elif i >= max_iters:
            warnings.warn("Maximum iterations reached")
            return U, i, diff
