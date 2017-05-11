#from gridtools import uniform_grid
from grids import FDTriMesh, FDRegularGrid
import numpy as np
import distmesh as dm
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay



# Regular grid on unit square
N = 9;
d = 2;
xi = [0,1]

shape = [N for i in range(d)]
bounds = np.array([xi for i in range(d)]).T
r = 2

Gu = FDRegularGrid(shape,bounds,r)

# Uniform grid on unit circle
fd = lambda p : np.sqrt((p**2).sum(1))-1.0
p, _ = dm.distmesh2d(fd, dm.huniform, 0.1, (-1,-1,1,1))

# delete boundary points generated by distmesh
cvx = ConvexHull(p)
boundary = cvx.vertices
p = p[np.in1d(np.arange(p.shape[0]),boundary,invert=True)]

# replace with a finer boundary resolution
th = np.linspace(0,2*np.pi,250)
th = th[0:-1]
p = np.concatenate([p,np.array([np.cos(th),np.sin(th)]).T])
boundary = np.arange(p.shape[0]-th.size,p.shape[0])
interior = np.arange(p.shape[0]-th.size)

# Get triangulation
dly = Delaunay(p)
t = dly.simplices

Gs = FDTriMesh(p, t,  boundary=boundary, interior = interior)
