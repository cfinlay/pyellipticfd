import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import itertools

from pyellipticfd.fdclasses import FDRegularGrid

# Regular grid on unit square
# ---------------------------
N = 9;
d = 2;
xi = [0,1]

shape = [N for i in range(d)]
bounds = np.array([xi for i in range(d)]).T
r = 2

Grid = FDRegularGrid(shape,bounds,r,colinear_tol=1e-4)
plt.plot(Grid.points[:,0], Grid.points[:,1],'ro',ms=.5)

# Example point and neighbours
i=8
ix = Grid.neighbours[Grid.neighbours[:,0]==i,1]
plt.plot(Grid.points[i,0],Grid.points[i,1],'bx')
plt.plot(Grid.points[ix,0],Grid.points[ix,1],'bo',ms=1)

ix = Grid.simplices[Grid.simplices[:,0]==i,1:]
spts = Grid.points[ix]
for pair in spts:
    plt.plot(pair[:,0],pair[:,1],'k',linewidth=.5)

plt.show()
