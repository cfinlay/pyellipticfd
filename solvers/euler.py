import numpy as np
import warnings
import itertools

def euler(U0,F,CFL,solution_tol=1e-4,max_iters=1e5):
    U = np.copy(U0)
    for i in itertools.count(1):
        U_interior = U[1:-1,1:-1] + CFL * F(U)

        diff = np.amax(np.absolute(U[1:-1,1:-1] - U_interior))
        U[1:-1,1:-1] = U_interior

        if diff < solution_tol:
            return U, i, diff
        elif i >= max_iters:
            warnings.warn("Maximum iterations reached")
            return U, i, diff
