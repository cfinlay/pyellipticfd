from gridtools import uniform_grid
from grids import FDGraph
import numpy as np
import distmesh as dm
import itertools



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
p, t = dm.distmesh2d(fd, dm.huniform, 0.05, (-1,-1,1,1))

edges = [np.array([v1,v2]).transpose() for v1,v2 in itertools.permutations(t.transpose(),2)]
edges = np.concatenate(edges)

(pc, r) = dm.circumcenter(p, t)
resolution = r.max()
eb = dm.boundedges(p, t)
boundary = np.unique(eb.flatten())

Gs = FDGraph(p, boundary=boundary, resolution = resolution,
        edges=edges)

