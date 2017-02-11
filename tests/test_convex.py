from context import solvers, utils
from solvers import convex_envelope
from utils import plot_utils
from IPython import get_ipython
import matplotlib.pyplot as plt
import numpy as np

ipython = get_ipython()

plt.close('all')

# Set up computational domain 
Nx = 41                      #grid size
dx = 2./(Nx-1)               #grid resolution
x = np.linspace(-1.,1.,num = Nx) #gridpoints in x coordinate
X, Y = np.meshgrid(x,x,sparse=False,indexing='ij') # create x & y gridpoints, with matrix indexing

#Pick an obstacle
#G = X**2 + Y**2
#G  = np.minimum(abs(X-0.5),abs(X+0.5))
#G = abs(np.sin(X**2+Y**2)-0.5)
#G = abs(np.sin(Y*np.pi)+np.cos(X*np.pi))
a = (-1/3,-2/3)
b = (1/3,2/3)
G = abs(np.minimum(np.sqrt((X+a[0])**2+(Y+a[1])**2),
                   np.sqrt((X+b[0])**2+(Y+b[1])**2)))


print("Euler step, on grid:")
ipython.magic("timeit convex_envelope.euler_step(G,dx,max_iters=1e5, method='grid')")
Eul_grid = convex_envelope.euler_step(G,dx,max_iters=1e5, method='grid')
#utils.plot_utils.plotter3d(X,Y,Eul_grid[0])
#plt.figure()
#plt.contour(X, Y, Eul_grid[0])

print("\nEuler step, interpolating:")
ipython.magic("timeit convex_envelope.euler_step(G,dx,max_iters=1e5, method='interpolate')")
Eul_intp = convex_envelope.euler_step(G,dx,max_iters=1e5, method='interpolate')
#utils.plot_utils.plotter3d(X,Y,Eul_intp[0])
#plt.figure()
#plt.contour(X, Y, Eul_intp[0])

print("\nPolicy iteration, on grid")
ipython.magic("timeit convex_envelope.policy(G,dx,max_iters=1e5,max_euler_iters=15, method='grid')")
Pol_grid = convex_envelope.policy(G,dx,max_iters=1e5,max_euler_iters=15,method='grid')
#utils.plot_utils.plotter3d(X,Y,Pol_grid[0])
#plt.figure()
#plt.contour(X, Y, Pol[0])

print("\nPolicy iteration, interpolating")
ipython.magic("timeit convex_envelope.policy(G,dx,max_iters=1e5,max_euler_iters=15, method='interpolate')")
Pol_intp = convex_envelope.policy(G,dx,max_iters=1e5,max_euler_iters=15, method='interpolate')
#utils.plot_utils.plotter3d(X,Y,Pol_intp[0])
#plt.figure()
#plt.contour(X, Y, Pol[0])
