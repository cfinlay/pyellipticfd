import warnings
import numpy as np
import time
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve, lsmr

from pyellipticfd import ddi, ddg, solvers, laplace

def solve(Grid,f=None,dirichlet=None,neumann=None,U0=None,fdmethod='interpolate',
          solver="direct",fredholm_tol = None,**kwargs):
    r"""
    Solve either the Poisson equation with Dirichlet BC
    \[
        -\Delta u = f, \,x \in \Omega \\
                u = g, \,x \in \partal \Omega,
    \]
    or the Poisson problem with Neumann BC
    \[
        -\Delta u = f, \,x \in \Omega \\
        \frac{\partial u}{\partial n} = h,\, x \in \partal \Omega.
    \]

    If f is not specified, solve the Laplace's equation (with f=0).

    Parameters
    ----------
    Grid : FDPointCloud
        The mesh of grid points.
    f : array_like or function
        The forcing function on the interior. Defaults to zero.
    dirichlet : array_like or function
        Dirichlet boundary condition.
    neumann : array_like or function
        Neumann boundary condition.
    U0 : array_like
        Initial guess. Optional. Only used if solver is 'euler'.
    fdmethod : string
        Which finite difference method to use.
        Either 'interpolate', 'grid', or 'froese'.
    solver : string
        Which solver to use. Either 'euler' or 'direct'.
    fredholm_tol : scalar
        Threshold for determining if the Fredholm alternative is satisfied.
        If set to None, no check is performed.
    **kwargs
        Additional arguments to be passed to the solver.

    Returns
    -------
    U : array_like
        The solution.
    time : scalar
        CPU time spent computing solution.

    Notes
    -----
    The parameters f, g, and h may be either numpy arrays, or functions. If f, g
    or h are functions, they take in a (N, dim) array and return (N,) arrays.
    """
    g = dirichlet
    h = neumann

    if g is None and h is None:
        raise ValueError('Please specify a boundary condition')
    elif g is not None and h is not None:
        raise ValueError('Cannot have both Dirichet and Neumann boundary conditions')

    if f is None:
        f = lambda x : np.zeros(x.shape[0])

    if callable(f):
        f = f(Grid.interior_points)

    if callable(g):
        g = g(Grid.bdry_points)

    if callable(h):
        h = h(Grid.bdry_points)


    # Construct the finite difference operators
    _, Lap = laplace.operator(Grid,fdmethod=fdmethod)

    if h is not None:
        if fdmethod=='interpolate':
            _, d1n = ddi.d1(Grid, -Grid.bdry_normals, domain='boundary')
        elif fdmethod=='grid':
            _, d1n = ddg.d1(Grid, -Grid.bdry_normals, domain='boundary')
        elif fdmethod=='froese':
            raise ValueError("Neumann boundary conditions not supported for Froese's finite difference method")

    # Finite difference matrix over the whole domain
    Jac = sparse.eye(Grid.num_points,format='csr')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Jac[Grid.interior]=-Lap
    if h is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Jac[Grid.bdry] = d1n

    # Forcing function over the whole domain
    F = np.zeros(Grid.num_points)
    if h is not None:
        F[Grid.bdry] = h
    else:
        F[Grid.bdry] = g
    F[Grid.interior] = f

    # Check that the system is consistent
    if fredholm_tol is not None:
        rho = spsolve(Jac.transpose(),np.zeros(Grid.num_points))
        rdotF =rho.dot(F)
        if rdotF > fredholm_tol:
            err = ("The problem is ill posed."
                    "\nForcing function does not satisfy the Fredholm alternative")
            raise ValueError(err)
    # TODO : consistency check for Neumann BC

    if solver=="euler":
        dt = 1/np.max(np.abs(Jac.diagonal())) # CFL condition

        # Initial guess
        if U0 is None and g is not None:
            U0 = np.ones(Grid.num_points)
            U0[Grid.bdry] = g
        elif U0 is None:
            U0 = np.ones(Grid.num_points)

        # Operator over whole domain
        def G(W):
            return Jac.dot(W) - F, dt

        if h is not None:
            # With Neumann conditions, return the solution with zero mean
            # (where each point has equal weight)
            U, diff, i, t = solvers.euler(U0, G, zeromean=True,**kwargs)
        else:
            U, diff, i, t = solvers.euler(U0, G, **kwargs)

        return U, t

    elif solver=="direct":
        t0 = time.time()

        if h is not None:
            # With Neumann BC, choose the solution with smallest L2 norm
            opt = lsmr(Jac, F, **kwargs)
            U = opt[0]
        else:
            U = spsolve(Jac,F, **kwargs)

        return U, time.time() - t0
