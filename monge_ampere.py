import warnings
import numpy as np
import time
import scipy.sparse as sparse

from pyellipticfd import ddi, ddg, solvers

def operator(Grid,W,jacobian=True,fdmethod='interpolate'):
    """
    Return the (convex) Monge Ampere operator on arbitrary grids.

    Parameters
    ----------
    Grid : FDPointCloud
        The mesh of grid points.
    W : array_like
        The function values.
    jacobian : boolean
        Whether to return the Jacobian, at W.
    fdmethod : string
        Which finite difference method to use. Either 'interpolate' or 'grid'.

    Returns
    -------
    val : array_like
        The operator value on the interior of the domain.
    M : scipy csr_matrix
        The Jacobian matrix of the operator at W.
    dt : scalar
        The CFL condition
    """
    if Grid.dim==3:
        raise NotImplementedError("Monge Ampere not yet implemented in 3d.")

    if fdmethod=='grid':
        op = ddg.d2eigs(Grid,W,jacobian=jacobian,control=False)
    elif fdmethod=='interpolate':
        op = ddi.d2eigs(Grid,W,jacobian=jacobian,control=False)

    (lmin, Mmin, _), (lmax, Mmax, _) = op

    bmin = lmin < 0
    bmax = lmax < 0
    lmin[bmin] = 0
    lmax[bmax] = 0
    FW = lmin*lmax

    #dx = Grid._VDist.min()
    dx = np.max([Grid.min_radius,Grid.min_edge_length])
    dt = 1/2 * dx**2 * 1/(lmin+lmax).max()

    if jacobian:
        Jac = sparse.diags(lmin).dot(Mmax) + sparse.diags(lmax).dot(Mmin)
    else:
        Jac = None

    return FW, Jac, dt

def solve(Grid,f,g,U0=None,fdmethod='interpolate', solver='newton',**kwargs):
    r"""
    Solve the Monge-Ampere equation,
    \[
        -\det [D^2u] = f, \,x \in \Omega \\
                   u = g, \,x \in \partal \Omega
                   u convex.
    \]

    Parameters
    ----------
    Grid : FDPointCloud
        The mesh of grid points.
    f : array_like or function
        Forcing function on interior.
    g : array_like or function
        Dirichlet boundary condition.
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
    The parameters f and g may be either numpy arrays or functions. If functions,
    they in (N, dim) arrays and return (N,) array.
    """

    if callable(f):
        f = f(Grid.interior_points)

    if callable(g):
        g = g(Grid.bdry_points)

    # Initial guess
    if U0 is None:
        U0 = np.zeros(Grid.num_points)
        U0[Grid.bdry] = g

    # Forcing function over the whole domain
    F = np.zeros(Grid.num_points)
    F[Grid.bdry] = -g
    F[Grid.interior] = f

    # Define the operator on the whole domain
    def G(W, jacobian=True):
        MA, M, dt = operator(Grid, W, jacobian=jacobian, fdmethod=fdmethod)

        GW = np.zeros(Grid.num_points)
        GW[Grid.interior] = -MA
        GW[Grid.bdry] = -W[Grid.bdry]

        if jacobian:
            Jac = sparse.diags(np.full(Grid.num_points,-1.0),format='csr')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Jac[Grid.interior]=-M
        else:
            Jac=None

        return (GW - F), Jac, dt


    if solver=="euler":
        def G_(W):
            op = G(W,jacobian=False)
            return op[0], op[2]

        return solvers.euler(U0, G_, **kwargs)
    elif solver=="newton":
        return solvers.newton(U0, G, scipysolver='lsmr',euler_ratio=1,**kwargs)
