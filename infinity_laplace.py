import warnings
import numpy as np
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
        Whether to calculate the finite difference matrix.
    fdmethod : string
        Which finite difference method to use. Either 'interpolate' or 'grid'.

    Returns
    -------
    val : array_like
        The operator value on the interior of the domain.
    M : scipy csr_matrix
        The finite difference matrix of the operator. Set to None if
        jacobian=False
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

    vpn = np.linalg.norm(P[2],axis=1)
    vmn = np.linalg.norm(M[2],axis=1)

    scaling = 2/(vpn + vmn)
    val = scaling*(d1p + d1m)

    if  jacobian:
        Dm, Dp = M[1], P[1]
        M = sparse.diags(scaling).dot(Dm+Dp)
    else:
        M = None

    return val, M

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
    The parameters f, dirichlet, and neumann may be either numpy arrays, or functions. If
    functions, they take in a (N, dim) array and return (N,) arrays.
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
        g = g(Grid.bdry_points)

    if callable(h):
        h = h(Grid.bdry_points)

    if h is not None:
        if fdmethod=='interpolate':
            d1n = ddi.d1(Grid, -Grid.bdry_normals, domain='boundary')
        elif fdmethod=='grid':
            d1n = ddg.d1(Grid, -Grid.bdry_normals, domain='boundary')

    # Forcing function over the whole domain
    F = np.zeros(Grid.num_points)
    if h is not None:
        F[Grid.bdry] = h
    else:
        F[Grid.bdry] = g
    F[Grid.interior] = f

    dx = np.max([Grid.min_radius,Grid.min_interior_nb_dist])
    dt = 1/2*dx**2 # CFL condition

    # Operator over whole domain
    def G(W, jacobian=True):

        inf_Lap, M = operator(Grid, W, jacobian=jacobian,fdmethod=fdmethod)

        GW = np.zeros(Grid.num_points)
        GW[Grid.interior] = -inf_Lap

        if h is not None:
            GW[Grid.bdry] = d1n.dot(W)
        else:
            GW[Grid.bdry] = W[Grid.bdry]

        if not jacobian:
            Jac = None
        else:
            # Fintite difference matrix
            Jac = sparse.eye(Grid.num_points,format='csr')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Jac[Grid.interior]=-M
            if h is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    Jac[Grid.bdry] = d1n

        return GW - F, Jac, dt


    # Initial guess
    if U0 is None and g is not None:
        U0 = np.ones(Grid.num_points)
        U0[Grid.bdry] = g
    elif U0 is None:
        U0 = np.ones(Grid.num_points)

    if solver=="euler":
        # Euler solver doesn't need jacobians
        def G_(W):
            op = G(W, jacobian=False)
            return op[0], op[2]

        if h is not None:
            # With Neumann conditions, return the solution with zero mean
            # (where each point has equal weight)
            return solvers.euler(U0, G_, zeromean=True,**kwargs)
        else:
            return solvers.euler(U0, G_, **kwargs)

    elif solver=="newton":
        return solvers.NewtonEulerLS(U0, G, **kwargs)
