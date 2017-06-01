import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import itertools

from fdclasses import FDRegularGrid



# Regular grid on unit square
# ---------------------------
N = 9;
d = 2;
xi = [0,1]

shape = [N for i in range(d)]
bounds = np.array([xi for i in range(d)]).T
r = 2

Gu = FDRegularGrid(shape,bounds,r,colinear_tol=1e-4)
plt.plot(Gu.points[:,0], Gu.points[:,1],'ro',ms=.5)

# Example point and neighbours
i=8
ix = Gu.neighbours[Gu.neighbours[:,0]==i,1]
plt.plot(Gu.points[i,0],Gu.points[i,1],'bx')
plt.plot(Gu.points[ix,0],Gu.points[ix,1],'bo',ms=1)

ix = Gu.simplices[Gu.simplices[:,0]==i,1:]
spts = Gu.points[ix]
for pair in spts:
    plt.plot(pair[:,0],pair[:,1],'k',linewidth=.5)

plt.show()
