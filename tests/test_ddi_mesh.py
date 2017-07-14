import time
import matplotlib.pyplot as plt
import numpy as np

from pyellipticfd import ddi
from setup import discs


# Test on circular domain for boundary normals
# --------------------------------------------
Grid, plot_sol = discs(2)

# Setup test functions
X, Y = Grid.points.T
U1 = X - Y
U2 = X**2 - Y**2
Un = np.linalg.norm(Grid.points, axis=1)

d1, M1 = ddi.d1(Grid,[1,0],U1, jacobian=True)

dgrad, Mgrad, vgrad = ddi.d1grad(Grid,U1,jacobian=True,control=True)

d1n, M_d1n = ddi.d1n(Grid, u=Un, jacobian=True)

d2, M2 = ddi.d2(Grid,[1,0],U2, jacobian=True)

t0 = time.time()
(d2min, M_min, v_min), (d2max, M_max, v_max) = ddi.d2eigs(Grid,U2, jacobian=True, control=True)
t1 = time.time()-t0

t0 = time.time()
(d2min, M_min, v_min), (d2max, M_max, v_max) = ddi.d2eigs(Grid,U2, jacobian=True, control=True)
t2 = time.time()-t0


## Plot stencil with worst error
## ----------------------------
#plt.plot(Grid.points[:,0], Grid.points[:,1],'ro',ms=.5)
#i = Grid.interior[d2min.argmax()]
#
#ix = Grid.neighbours[Grid.neighbours[:,0]==i,1]
#plt.plot(Grid.points[i,0],Grid.points[i,1],'bx')
#plt.plot(Grid.points[ix,0],Grid.points[ix,1],'bo',ms=1)
#
#ix = Grid.simplices[Grid.simplices[:,0]==i,1:]
#spts = Grid.points[ix]
#for pair in spts:
#    plt.plot(pair[:,0],pair[:,1],'k',linewidth=.5)
#
#X,Y = Grid.bdry_points.T
#plt.plot(X,Y,'ko',ms='.5')
#
#plt.show()
