import warnings
import numpy as np
import time
import scipy.sparse as sparse

from pyellipticfd import ddi, ddg, solvers

def operator(Grid,W,g,jacobian=True,fdmethod='interpolate'):
    """
    Return the convex envelope operator on arbitrary grids.

    Parameters
    ----------
    Grid : FDPointCloud
        The mesh of grid points.
    W : array_like
        The function values.
    g : array_like
        The obstacle
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

    if fdmethod=='grid':
        op = ddg.d2min(Grid,W,jacobian=jacobian,control=False)
    elif fdmethod=='interpolate':
        op = ddi.d2min(Grid,W,jacobian=jacobian,control=False)

    if jacobian:
        lambda1, M = op
    else:
        lambda1 = op

    FW = W-g
    b = -lambda1 > FW[Grid.interior]
    FW[Grid.interior[b]] = -lambda1[b]

    if jacobian:
        Jac = sparse.eye(Grid.num_points,format='csr')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") #suppress stupid warning
            Jac[Grid.interior[b],:] = -M[b,:]

        return FW, Jac
    else:
        return FW

def solve(Grid,g,U0=None,fdmethod='interpolate', solver="newton",**kwargs):
    r"""
    Find the convex envelope of the obstacle g(x), via the
    solution of the PDE:
    \[
        -\max\{-\lambda_1[u], u-g \} = 0, \,x \in \Omega \\
                                u = g, \,x \in \partal \Omega.
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
    The parameter g may be either an numpy array or a functions. If g
    is a function, it takes in a (N, dim) array and returns  an (N,) array.
    """

    if callable(g):
        g = g(Grid.points)
        gb = g[Grid.boundary]
    else:
        gb = g[Grid.boundary]

    # Initial guess
    if U0 is None:
        U0 = np.copy(g)

    dt = 1/2*np.max([Grid.min_edge_length, Grid.min_radius])**2 # CFL condition

    if solver=="euler":
        return solvers.euler(U0,
                             lambda W : operator(Grid, W, g,
                                                 jacobian=False, fdmethod=fdmethod),
                             dt, **kwargs)

    elif solver=="newton":
        return solvers.newton(U0,
                             lambda W, jacobian=True : operator(Grid, W, g,
                                                 jacobian=jacobian, fdmethod=fdmethod),
                              dt, **kwargs)
