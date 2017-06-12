import warnings
import numpy as np
import time
import scipy.sparse as sparse

from pyellipticfd import ddi, ddg, solvers

def solve(Grid,g,U0=None,fdmethod='interpolate', solver="newton",**kwargs):
    r"""
    Find the convex envelope of the obstacle g(x), via the
    solution of the PDE:
    \[
        -\max\{-\lambda_1[u], g \} = 0, \,x \in \Omega \\
                                u = g, \,x \in \partal \Omega,
    \]

    Parameters
    ----------
    Grid : FDPointCloud
        The mesh of grid points.
    g : array_like or function
        The obstacle.
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

    Notes
    -----
    The parametres f, g, and h may be either numpy arrays, or functions. If f, g
    or h are functions, they take in a (N, dim) array and return (N,) arrays.
    """
    g = dirichlet
    h = neumann

    if g is None and h is None:
        raise ValueError('Please specify a boundary condition')
    elif g is not None and h is not None:
        raise ValueError('Cannot have both Dirichet and Neumann boundary conditions')

    if callable(f):
        f = f(Grid.interior_points)

    if callable(g):
        g = g(Grid.boundary_points)

    if callable(h):
        h = h(Grid.boundary_points)

    if h is not None:
        if fdmethod=='interpolate':
            d1n = ddi.d1(Grid, -Grid.boundary_normals, domain='boundary')
        elif fdmethod=='grid':
            d1n = ddg.d1(Grid, -Grid.boundary_normals, domain='boundary')

    # Forcing function over the whole domain
    F = np.zeros(Grid.num_points)
    if h is not None:
        F[Grid.boundary] = h
    else:
        F[Grid.boundary] = g
    F[Grid.interior] = f
