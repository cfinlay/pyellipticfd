import warnings
import numpy as np
import scipy.sparse as sparse

from pyellipticfd import ddi, ddg, solvers

def solve(Grid,f,dirichlet=None,neumann=None,U0=None,fdmethod='interpolate',
          solver="direct",**kwargs):
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

    Parameters
    ----------
    Grid : FDPointCloud
        The mesh of grid points.
    f : array_like or function
        The forcing function on the interior.
    dirichlet : array_like or function
        Dirichlet boundary condition.
    neumann : array_like or function
        Neumann boundary condition
    U0 : array_like
        Initial guess. Optional.
    fdmethod : string
        Which finite difference method to use. Either 'interpolate' or 'grid'.
    solver : string
        Which solver to use. Either 'euler' or 'direct'.
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

    if U0 is None and g is not None:
        U0 = np.ones(Grid.num_points)
        U0[Grid.boundary] = g
    elif U0 is None:
        U0 = np.ones(Grid.num_points)


    if fdmethod=='interpolate':
        D2 = [ddi.d2(Grid,e) for e in np.identity(Grid.dim)]
    elif fdmethod=='grid':
        D2 = [ddg.d2(Grid,e) for e in np.identity(Grid.dim)]

    Lap = np.sum(D2)


    Grad = sparse.eye(Grid.num_points,format='csr')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Grad[Grid.interior]=-Lap

    if h is not None:
        if fdmethod=='interpolate':
            d1n = ddi.d1(Grid, -Grid.boundary_normals, domain='boundary')
        if fdmethod=='grid':
            d1n = ddg.d1(Grid, -Grid.boundary_normals, domain='boundary')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Grad[Grid.boundary] = d1n

    dt = 1/np.max(np.abs(Grad.diagonal()))

    def F(W,jacobian=True):
        LW = -Lap.dot(W) - f

        if g is not None:
            HW = W[Grid.boundary] - g
        elif h is not None:
            HW = d1n.dot(W) - h

        FW = np.zeros(Grid.num_points)
        FW[Grid.interior] = LW
        FW[Grid.boundary] = HW

        if not jacobian:
            return FW
        else:
            return FW, Grad


    if solver=="euler":
        if h is not None:
            zeromean = True
        else:
            zeromean = False
        return solvers.euler(U0, lambda W: F(W,jacobian=False), dt, zeromean=zeromean,**kwargs)
    elif solver=="direct":
        return solvers.newton(U0, F, dt, **kwargs)
