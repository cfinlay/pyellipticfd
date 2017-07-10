import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import itertools

from pyellipticfd.fdclasses import FDRegularGrid
from pyellipticfd import ddi

# Regular grid on unit square
# ---------------------------
N = 19;
d = 2;
xi = [0,1]

shape = [N for i in range(d)]
bounds = np.array([xi for i in range(d)]).T
r = 4

Grid = FDRegularGrid(shape,bounds,r)
plt.plot(Grid.points[:,0], Grid.points[:,1],'ro',ms=.5)

X, Y = Grid.points.T
th = np.pi/3
X_ = np.cos(th)*X + np.sin(th)*Y
Y_ = -np.sin(th)*X + np.cos(th)*Y
U2 = X_**2 + Y_**2

d2, _ = ddi.d2(Grid,[1,0],U2)
i = Grid.interior[d2.argmax()]
#d1n, _ = ddi.d1n(Grid,U2)
#i = d1n.argmin()
#i = Grid.boundary[i]

ix = Grid.neighbours[Grid.neighbours[:,0]==i,1]
plt.plot(Grid.points[i,0],Grid.points[i,1],'bx')
plt.plot(Grid.points[ix,0],Grid.points[ix,1],'bo',ms=1)

ix = Grid.simplices[Grid.simplices[:,0]==i,1:]
spts = Grid.points[ix]
for pair in spts:
    plt.plot(pair[:,0],pair[:,1],'k',linewidth=.5)

X,Y = Grid.bdry_points.T
plt.plot(X,Y,'ko',ms='.5')

plt.show()
