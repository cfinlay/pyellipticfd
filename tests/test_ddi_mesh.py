import numpy as np
import distmesh as dm
from scipy.spatial import ConvexHull, Delaunay

from pyellipticfd import ddi


# Test on circular domain for boundary normals
# --------------------------------------------
from setup import discs
Grid, plot_sol = discs(0)

X, Y = Grid.points.T
U2 = X**2 + Y**2

U = np.linalg.norm(Grid.points, axis=1)
d2, M2 = ddi.d2(Grid,[1,0],U2, jacobian=True)
(d2min, M_min, v_min), (d2max, M_max, v_max) = ddi.d2eigs(Grid,U2, jacobian=True, control=True)

d1n, M_d1n = ddi.d1n(Grid, u=U, jacobian=True)
