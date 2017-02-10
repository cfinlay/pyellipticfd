from context import solvers, utils
from solvers import convex_envelope as ce
from utils import plot_utils
import matplotlib.pyplot as plt

import numpy as np
plt.close('all')
# Set up computational domain 
Nx = 2**6                      #grid size
dx = 2./(Nx-1)               #grid resolution
x = np.linspace(-1,1.,num = Nx) #gridpoints in x coordinate
X, Y = np.meshgrid(x,x,sparse=False,indexing='ij') # create x & y gridpoints, with matrix indexing

#Pick an obstacle
#G = X**2 + Y**2
#G  = np.minimum(abs(X-0.5),abs(X+0.5))
#G = abs(np.sin(X**2+Y**2)-0.5)
#G = abs(np.sin(Y*np.pi)+np.cos(X*np.pi))
G = abs(np.minimum(np.sqrt((X-0.5)**2+(Y+1/6)**2),
                   np.sqrt((X+0.5)**2+(Y-np.pi/5)**2)))

U = ce.euler_step(G,dx,max_iters=10000)
plot_utils.plotter3d(X,Y,G)

plot_utils.plotter3d(X,Y,U)

plt.figure()
plt.contour(X, Y, U)