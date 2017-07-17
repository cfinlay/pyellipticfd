import numpy as np
from scipy.spatial.distance import pdist, squareform

from pyellipticfd.pointclasses import FDRegularGrid
from pyellipticfd import ddg

def get_grid(angle):
    # Regular grid on unit square
    # ---------------------------
    N = 19;
    d = 2;
    xi = [0,1]

    shape = [N for i in range(d)]
    bounds = np.array([xi for i in range(d)]).T
    r = 4

    Grid = FDRegularGrid(shape,bounds,r,interpolation=False)


    # Test function
    # -------------
    X, Y = Grid.points.T
    X_ = np.cos(angle)*X + np.sin(angle)*Y
    Y_ = -np.sin(angle)*X + np.cos(angle)*Y
    U2 = X_**2 - Y_**2

    return Grid, U2


# These should be exact if the grid is setup properly
def test_exact_derivatives():
    Grid, U2 = get_grid(0)
    d2x, _ = ddg.d2(Grid,[1,0],U2)
    d2y, _ = ddg.d2(Grid,[0,1],U2)
    assert np.abs(d2x.max()) - 2 < 1e-13
    assert np.abs(d2y.max()) - 2 < 1e-13
