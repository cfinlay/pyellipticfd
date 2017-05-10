#from gridtools import uniform_grid
from grids import FDTriMesh
import numpy as np
import distmesh as dm
from scipy.spatial import ConvexHull



## Uniform grid on unit square
#N = 2**4+1;
#d = 2;
#xi = [0,1]
#
#shape = [N for i in range(d)]
#bounds = np.array([xi for i in range(d)]).T
#r = 2
#
#Gu = uniform_grid(shape,bounds,r)

# Uniform grid on unit circle
fd = lambda p : np.sqrt((p**2).sum(1))-1.0
p, t = dm.distmesh2d(fd, dm.huniform, 0.1, (-1,-1,1,1))

cvx = ConvexHull(p)
boundary = cvx.vertices

Gs = FDTriMesh(p, t,  boundary=boundary)
