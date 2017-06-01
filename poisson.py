import warnings
import numpy as np
import scipy.sparse as sparse

from pyellipticfd import ddi, ddg, solvers

def solve(Grid,f,g,U0=None,fdmethod='interpolate',
          solver="euler",**kwargs):
    """
    Solve the Poisson equation
    \[
        -\Delta u = f, x \in \Omega
                u = g, x \in \partal \Omega
    \]

    Parameters
    ----------
    Grid : FDPointCloud
        The mesh of grid points.
    f : array_like or function
        The forcing function on the interior.
        Either a numpy array the same length as the number of interior points,
        or a function.
    g : array_like or function
        Dirichlet boundary condition.
        Either a numpy array the same length as the number of interior points,
        or a function.
    U0 : array_like
        Initial guess. Optional.
    fdmethod : string
        Which finite difference method to use. Either 'interpolate' or 'grid'.
    solver : string
        Which solver to use. Either 'euler' or 'newton'.
    **kwargs
        Additional arguments to be passed to the solver.

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

    def F(W,jacobian=True):
        LW = -Lap.dot(W) - f
        Wb = W[Grid.boundary] - g

        FW = np.zeros(Grid.num_points)
        FW[Grid.interior] = LW
        FW[Grid.boundary] = Wb

        if not jacobian:
            return FW
        else:
            Grad = sparse.eye(Grid.num_points,format='csr')
            # TODO: deal with sparsity structure warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Grad[Grid.interior]=-Lap

            return FW, Grad


    if solver=="euler":
        return solvers.euler(U0, lambda W: F(W,jacobian=False), dt, **kwargs)
    elif solver=="newton":
        return solvers.newton(U0, F, dt, **kwargs)
