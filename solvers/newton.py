import itertools
import numpy as np
from scipy.sparse.linalg import spsolve
import warnings

def newton(U,operator,solution_tol=1e-4,max_iters=1e2):
    """
    Use semismooth Newton's method to find the steady state F[U]=0.
    """
    Nx, Ny = U.shape

    for i in itertools.count(1):
        Fu, Grad = operator(U)
        d = spsolve(Grad,-Fu)
        U_interior = U[1:-1,1:-1] + np.reshape(d,(Nx-2,Ny-2))

        diff = np.amax(np.absolute(U[1:-1,1:-1] - U_interior))
        U[1:-1,1:-1] = U_interior

        if diff < solution_tol:
            return U, i, diff
        elif i >= max_iters:
            warnings.warn("Maximum iterations reached")
            return U, i, diff

