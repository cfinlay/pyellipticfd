import itertools
import numpy as np
from scipy.sparse.linalg import spsolve
import warnings

def newton(U,operator,solution_tol=1e-4,max_iters=1e2):
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
        Fu, Grad = operator(U)
        d = spsolve(Grad,-Fu)
        
        U_new = U + np.reshape(d,U.shape)
        diff = np.amax(np.absolute(U - U_new))
        U = U_new

        if diff < solution_tol:
            return U, i, diff
        elif i >= max_iters:
            warnings.warn("Maximum iterations reached")
            return U, i, diff

