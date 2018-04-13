import warnings
import numpy as np
import scipy.sparse as sparse

from pyellipticfd import ddi, ddg, ddf, solvers

def operator(Grid, W, A, jacobian=True,fdmethod='interpolate'):
    """
    Return the finite difference Pucci operator on arbitrary grids.

    Parameters
    ----------
    Grid : FDPointCloud
        The mesh of grid points.
    U : array_like
        The function values.
    A : list
        A list of numpy arrays, or scalars. The arrays correspond to the spatial coefficients
        of the Hessian's eigenvalues. Ordered from maximal to minimal eigenvalues.
    jacobian : boolean
        Whether to calculate the finite difference matrix.
    fdmethod : string
        Which finite difference method to use. Either 'interpolate', 'grid' or 'froese'.

    Returns
    -------
    val : array_like
        The operator value on the interior of the domain.
    M : scipy csr_matrix
        The finite difference matrix of the operator. Set to None if
        jacobian=False
    """
    if Grid.dim==3:
        raise NotImplementedError("Pucci operator not yet implemented in 3d.")

    # Construct the finite difference operator, get value of first
    # derivative in minimal and maximal gradient directions
    if fdmethod=='grid':
        op = ddg.d2eigs(Grid,W,jacobian=jacobian,control=False)
    elif fdmethod=='interpolate':
        op = ddi.d2eigs(Grid,W,jacobian=jacobian,control=False)
    elif fdmethod=='froese':
        op = ddf.d2eigs(Grid,W,jacobian=jacobian,control=False)

    (lmin, Mmin, _), (lmax, Mmax, _) = op


    val = A[0]*lmax + A[1]*lmin

    if jacobian:
        if np.isscalar(A[0]) and np.isscalar(A[1]):
            M = A[0]*Mmax + A[1]*Mmin
        else:
            M = sparse.diags(A[0]).dot(Mmax) + sparse.diags(A[1]).dot(Mmin)
    else:
        M = None

    return -val, -M

def solve(Grid,f,dirichlet,A=(2,1),U0=None,fdmethod='interpolate',
          solver="newton",**kwargs):
    r"""
    Solve the Pucci equation with Dirichlet BC
    \[
        A[0] \lambda_1[u] + A[2]\lambda_2[u] = f, \,x \in \Omega \\
                       u = g, \,x \in \partal \Omega
    \]

    Parameters
    ----------
    Grid : FDPointCloud
        The mesh of grid points.
    f : array_like or function
        The forcing function on the interior.
    dirichlet : array_like or function
        Dirichlet boundary condition.
    A : tuple of scalars or array_like
        Coefficient of maximal eigenvalue
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
    The parameters f anddirichlet may be either numpy arrays, or functions. If
    functions, they take in a (N, dim) array and return (N,) arrays.
    """
    g = dirichlet

    if callable(f):
        f = f(Grid.interior_points)

    if callable(g):
        g = g(Grid.bdry_points)


    # Forcing function over the whole domain
    F = np.zeros(Grid.num_points)
    F[Grid.bdry] = g
    F[Grid.interior] = f

    dx = np.max([Grid.min_radius,Grid.min_interior_nb_dist])
    dt = 1/2*dx**2 # CFL condition

    # Operator over whole domain
    def G(W, jacobian=True):

        val, M = operator(Grid, W, A, jacobian=jacobian,fdmethod=fdmethod)

        GW = np.zeros(Grid.num_points)
        GW[Grid.interior] = -val

        GW[Grid.bdry] = W[Grid.bdry]

        if not jacobian:
            Jac = None
        else:
            # Fintite difference matrix
            Jac = sparse.eye(Grid.num_points,format='csr')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Jac[Grid.interior]=-M

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

        return solvers.euler(U0, G_, **kwargs)

    elif solver=="newton":
        return solvers.NewtonEulerLS(U0, G, **kwargs)
