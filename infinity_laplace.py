import warnings
import numpy as np
import time
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve, lsmr

from pyellipticfd import ddi, ddg, solvers

def solve(Grid,f,dirichlet=None,neumann=None,U0=None,fdmethod='interpolate',
          solver="newton",**kwargs):
    r"""
    Solve either the infinity Laplacian with Dirichlet BC
    \[
        -\Delta_\infty u = f, \,x \in \Omega \\
                       u = g, \,x \in \partal \Omega,
    \]
    or with Neumann BC
    \[
        -\Delta_infty u = f, \,x \in \Omega \\
        \frac{\partial u}{\partial n} = h,\, x \in \partal \Omega.
    \]

    Parameters
    ----------
    Grid : FDPointCloud
        The mesh of grid points.
    f : array_like or function
        The forcing function on the interior.
    dirichlet : array_like or function
        Dirichlet boundary condition.
    neumann : array_like or function
        Neumann boundary condition.
    U0 : array_like
        Initial guess. Optional. Only used if solver is 'euler'.
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

    # Operator over whole domain
    def G(W, jacobian=True):
        if fdmethod=='interpolate':
            P = ddi.d1grad(Grid,u,control=True,jacobian=jacobian)
            M = ddi.d1grad(Grid,-u, control=True,jacobian=jacobian)
        elif fdmethod=='grid':
            P = ddg.d1grad(Grid,u,control=True,jacobian=jacobian)
            M = ddg.d1grad(Grid,-u, control=True,jacobian=jacobian)
        d1p, d1m = P[0], -M[0]
        if jacobian is False:
            vpn = np.linalg.norm(P[1],axis=1)
            vmn = np.linalg.norm(M[1],axis=1)
        else:
            vpn = np.linalg.norm(P[2],axis=1)
            vmn = np.linalg.norm(M[2],axis=1)

        scaling = 2/(vpn + vmn)
        inf_Lap = scaling*(d1p + d1m)

        GW = np.zeros(Grid.num_points)
        GW[Grid.interior] = -inf_Lap
        GW[Grid.boundary] = d1n.dot(W)

        if jacobian is False:
            return GW - F
        else:
            Dm, Pm = M[1], P[1]

            # Fintite difference matrix
            Jac = sparse.eye(Grid.num_points,format='csr')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Jac[Grid.interior]=(-sparse.diags(scaling)).dot(Dm+Dp)
            if h is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    Jac[Grid.boundary] = d1n

            return GW - F, Jac


    if solver=="euler":
        dt = 1/(2*np.max(Grid.min_edge_length, Grid.min_radius)**2) # CFL condition

        # Initial guess
        if U0 is None and g is not None:
            U0 = np.ones(Grid.num_points)
            U0[Grid.boundary] = g
        elif U0 is None:
            U0 = np.ones(Grid.num_points)

        if h is not None:
            # With Neumann conditions, return the solution with zero mean
            # (where each point has equal weight)
            U, diff, i, t = solvers.euler(U0, G, dt, zeromean=True,**kwargs)
        else:
            U, diff, i, t = solvers.euler(U0, G, dt, **kwargs)

        return U, t

    elif solver=="newton":
        t0 = time.time()

        if h is not None:
            # With Neumann BC, choose the solution with smallest L2 norm
            return solvers.newton(U0, G, dt, solver='lsmr',**kwargs)
        else:
            return solvers.newton(U0, G, dt, **kwargs)
