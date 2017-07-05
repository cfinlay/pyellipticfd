import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import itertools

from pyellipticfd import ddi
from generate_grids import gen_grid

# Grid on unit disc
# -----------------
Grid  = gen_grid(h,h**(1/3)*np.pi)
plt.plot(Grid.points[:,0], Grid.points[:,1],'ro',ms=.5)

X, Y = Grid.points.T
U2 = X**2 + Y**2

d2, _ = ddi.d2(Grid,[1,0],U2)
i = d2.argmax()

ix = Grid.neighbours[Grid.neighbours[:,0]==i,1]
plt.plot(Grid.points[i,0],Grid.points[i,1],'bx')
plt.plot(Grid.points[ix,0],Grid.points[ix,1],'bo',ms=1)

ix = Grid.simplices[Grid.simplices[:,0]==i,1:]
spts = Grid.points[ix]
for pair in spts:
    plt.plot(pair[:,0],pair[:,1],'k',linewidth=.5)

X,Y = Grid.boundary_points.T
plt.plot(X,Y,'ko',ms='.5')

plt.show()
