import warnings
import numpy as np
import time
import scipy.sparse as sparse

from pyellipticfd import ddi, ddg, solvers

def operator(Grid, U, jacobian=True,fdmethod='interpolate'):
    """
    Return the finite difference infinity Laplace operator on arbitrary grids.

    Parameters
    ----------
    Grid : FDPointCloud
        The mesh of grid points.
    U : array_like
        The function values.
    jacobian : boolean
        Whether to return the finite difference matrix.
    fdmethod : string
        Which finite difference method to use. Either 'interpolate' or 'grid'.

    Returns
    -------
    val : array_like
        The operator value on the interior of the domain.
    M : scipy csr_matrix
        The finite difference matrix of the operator.
    """

    # Construct the finite difference operator, get value of first
    # derivative in minimal and maximal gradient directions
    if fdmethod=='interpolate':
        P = ddi.d1grad(Grid,U,control=True,jacobian=jacobian)
        M = ddi.d1grad(Grid,-U, control=True,jacobian=jacobian)
    elif fdmethod=='grid':
        P = ddg.d1grad(Grid,U,control=True,jacobian=jacobian)
        M = ddg.d1grad(Grid,-U, control=True,jacobian=jacobian)
    d1p, d1m = P[0], -M[0]

    # Norm of direction of minimal and maximal gradient
    if not jacobian:
        vpn = np.linalg.norm(P[1],axis=1) # maximal gradient direction
        vmn = np.linalg.norm(M[1],axis=1) # minimal gradient direction
    else:
        vpn = np.linalg.norm(P[2],axis=1)
        vmn = np.linalg.norm(M[2],axis=1)

    scaling = 2/(vpn + vmn)
    val = scaling*(d1p + d1m)

    if  jacobian:
        Dm, Dp = M[1], P[1]
        M = sparse.diags(scaling).dot(Dm+Dp)
        return val, M
    else:
        return val

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
    The parameters f, g, and h may be either numpy arrays, or functions. If f, g
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

        op = operator(Grid, W, jacobian=jacobian,fdmethod=fdmethod)

        if jacobian:
            inf_Lap, M = op
        else:
            inf_Lap = op

        GW = np.zeros(Grid.num_points)
        GW[Grid.interior] = -inf_Lap

        if h is not None:
            GW[Grid.boundary] = d1n.dot(W)
        else:
            GW[Grid.boundary] = W[Grid.boundary]

        if not jacobian:
            return GW - F
        else:
            # Fintite difference matrix
            Jac = sparse.eye(Grid.num_points,format='csr')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Jac[Grid.interior]=-M
            if h is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    Jac[Grid.boundary] = d1n

            return GW - F, Jac

    dt = 1/2*np.max([Grid.min_edge_length, Grid.min_radius])**2 # CFL condition

    # Initial guess
    if U0 is None and g is not None:
        U0 = np.ones(Grid.num_points)
        U0[Grid.boundary] = g
    elif U0 is None:
        U0 = np.ones(Grid.num_points)

    if solver=="euler":
        if h is not None:
            # With Neumann conditions, return the solution with zero mean
            # (where each point has equal weight)
            return solvers.euler(U0, lambda W : G(W, jacobian=False),
                                          dt, zeromean=True,**kwargs)
        else:
            return solvers.euler(U0, lambda W : G(W, jacobian=False),
                                          dt, **kwargs)

    elif solver=="newton":
        t0 = time.time()

        if h is not None:
            # With Neumann BC, choose the solution with smallest L2 norm
            return solvers.newton(U0, G, dt, solver='lsmr',**kwargs)
        else:
            return solvers.newton(U0, G, dt, **kwargs)
