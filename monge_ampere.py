import warnings
import numpy as np
import time
import scipy.sparse as sparse

from pyellipticfd import ddi, ddg, ddf, solvers

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
    elif fdmethod=='froese':
        op = ddf.d2eigs(Grid,W,jacobian=jacobian,control=False)

    (lmin, Mmin, _), (lmax, Mmax, _) = op

    bmin = lmin < 0
    bmax = lmax < 0
    lmin[bmin] = 0
    lmax[bmax] = 0
    FW = lmin*lmax

    #dx = Grid._VDist.min()
    dx = np.max([Grid.min_radius,Grid.min_interior_nb_dist])
    dt = 1/2 * dx**2 * 1/(lmin+lmax).max()

    if jacobian:
        Jac = sparse.diags(lmin).dot(Mmax) + sparse.diags(lmax).dot(Mmin)
    else:
        Jac = None

    return FW, Jac, dt

def solve(Grid,f,dirichlet=None,neumann=None,
        U0=None,fdmethod='interpolate', solver='euler',**kwargs):
    r"""
    Solve the Monge-Ampere equation, either with Dirichlet BC
    \[
        -\det [D^2u] = f, \,x \in \Omega \\
                   u = g, \,x \in \partal \Omega
                   u convex,
    \]
    or with Neumann BC
    \[
        -\det [D^2u] = f, \,x \in \Omega \\
        \frac{\partial u}{\partial n} = h,\, x \in \partal \Omega.
        u convex,
    where n is the outward normal.
    \]

    Parameters
    ----------
    Grid : FDPointCloud
        The mesh of grid points.
    f : array_like or function
        Forcing function on interior.
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

    if callable(g):
        g = g(Grid.bdry_points)

    if callable(f):
        f = f(Grid.interior_points)

    if callable(h):
        h = h(Grid.bdry_points)

    if h is not None:
        if fdmethod=='interpolate':
            d1n = ddi.d1(Grid, -Grid.bdry_normals, domain='boundary')[1]
        elif fdmethod=='grid' or fdmethod=='froese':
            d1n = ddg.d1(Grid, -Grid.bdry_normals, domain='boundary')[1]

    # Initial guess
    if U0 is None:
        U0 = np.einsum('ij,ij->i',Grid.points,Grid.points)
        if g is not None:
            U0 = U0-U0.max()+g.min()
            U0[Grid.bdry] = g
        else:
            U0 = U0-U0.max()

    # Forcing function over the whole domain
    F = np.zeros(Grid.num_points)
    if h is not None:
        F[Grid.bdry] = h
    else:
        F[Grid.bdry] = -g
    F[Grid.interior] = f

    if h is not None:
        dt_bdry = np.max([Grid.dist_to_bdry,Grid.min_radius])

    # Define the operator on the whole domain
    def G(W, jacobian=True):
        MA, M, dt = operator(Grid, W, jacobian=jacobian, fdmethod=fdmethod)

        GW = np.zeros(Grid.num_points)
        GW[Grid.interior] = -MA

        if h is not None:
            GW[Grid.bdry] = -d1n.dot(W)
        else:
            GW[Grid.bdry] = -W[Grid.bdry]

        if jacobian:
            Jac = sparse.diags(np.full(Grid.num_points,-1.0),format='csr')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Jac[Grid.interior]=-M
            if h is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    Jac[Grid.bdry] = -d1n
        else:
            Jac=None


        if h is not None:
            dt = np.min([dt, dt_bdry])


        return (GW - F), Jac, dt


    if solver=="euler":
        def G_(W):
            op = G(W,jacobian=False)
            return op[0], op[2]
        max_iters = 1/Grid.min_radius**2 * 50

        if h is None:
            return solvers.euler(U0, G_, **kwargs)
        else:
            return solvers.euler(U0, G_, zeromax=True,**kwargs)
    elif solver=="newton":
        return solvers.NewtonEulerLS(U0, G, euler_ratio=1,**kwargs)
