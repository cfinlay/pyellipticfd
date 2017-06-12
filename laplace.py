import numpy as np

from pyellipticfd import ddi, ddg

def operator(Grid, U=None, jacobian=True,fdmethod='interpolate'):
    """
    Return the finite difference Laplace operator on arbitrary grids.

    Parameters
    ----------
    Grid : FDPointCloud
        The mesh of grid points.
    U : array_like
        The function values. If not specified, only the finite difference
        matrix is returned.
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
    # Construct the finite difference operator
    if fdmethod=='interpolate':
        D2 = [ddi.d2(Grid,e) for e in np.identity(Grid.dim)]
    elif fdmethod=='grid':
        D2 = [ddg.d2(Grid,e) for e in np.identity(Grid.dim)]
    Lap = np.sum(D2)

    if U is not None and jacobian:
        return Lap.dot(U), Lap
    elif U is None:
        return Lap
    elif not jacobian:
        return Lap.dot(U)
