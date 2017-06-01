import numpy as np
from pyellipticfd import ddi, ddg, solvers

def solve(Grid,f,g,U0=None,fdmethod='interpolate'):
    """
    Solve the Poisson equation with forcing function f and
    Dirichlete boundary condition g.
    """
    if callable(f):
        f = f(Grid.interior_points)

    if callable(g):
        g = g(Grid.boundary_points)

    if U0 is None:
        U0 = np.ones(Grid.num_points)
        U0[Grid.boundary] = g


    if fdmethod=='interpolate':
        D2 = [ddi.d2(Grid,e) for e in np.identity(Grid.dim)]
    elif fdmethod=='grid':
        D2 = [ddg.d2(Grid,e) for e in np.identity(Grid.dim)]

    Lap = np.sum(D2)
    dt = -1/Lap.diagonal().min()

    def F(W):
        LW = -Lap.dot(W) - f
        Wb = W[Grid.boundary] - g

        FW = np.zeros(Grid.num_points)
        FW[Grid.interior] = LW
        FW[Grid.boundary] = Wb

        return FW

    return solvers.euler(U0, F, dt)
