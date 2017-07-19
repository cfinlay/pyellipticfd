import numpy as np
from scipy.spatial.distance import pdist, squareform
import itertools

from pyellipticfd import ddi
from setup_discs import disc_mesh

def get_grid():
    # Grid on unit disc
    # -----------------
    h = 0.03
    Grid  = disc_mesh(h,h**(1/3)*np.pi)

    X, Y = Grid.points.T
    U2 = X**2 + Y**2

    return Grid, U2

def test_derivatives():
    # These should be close to 2
    Grid, U2 = get_grid()
    d2x, _ = ddi.d2(Grid,[1,0],U2)
    d2y, _ = ddi.d2(Grid,[0,1],U2)
    assert d2x.max()-2 < 0.17
    assert d2y.max()-2 < 0.17

