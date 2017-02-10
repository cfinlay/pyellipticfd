from context import solvers, utils
from solvers import convex_envelope
from utils import plot_utils
from IPython import get_ipython

ipython = get_ipython()

import numpy as np

# Set up computational domain 
Nx = 41                      #grid size
dx = 2./(Nx-1)               #grid resolution
x = np.linspace(-1.,1.,num = Nx) #gridpoints in x coordinate
X, Y = np.meshgrid(x,x,sparse=False,indexing='ij') # create x & y gridpoints, with matrix indexing

# Define an obstacle
a = (-1/3,-2/3)
b = (1/3,2/3)
#a = (-1/3,0)
#b = (1/3,0)
def obstacle_fcn(X,Y):
    return np.minimum(np.sqrt((X-a[0])**2 + (Y-a[1])**2),
                      np.sqrt((X-b[0])**2 + (Y-b[1])**2))

# Calculate the obstacle and optionally plot it
G = obstacle_fcn(X,Y)

print("Euler step:")
ipython.magic("timeit Eul = convex_envelope.euler_step(G,dx,max_iters=1e5)")
Eul = convex_envelope.euler_step(G,dx,max_iters=1e5)
#utils.plot_utils.plotter3d(X,Y,Eul[0])

print("\nPolicy iteration")
ipython.magic("timeit Pol = convex_envelope.policy(G,dx,max_iters=1e5,max_euler_iters=30)")
Pol = convex_envelope.policy(G,dx,max_iters=1e5,max_euler_iters=30)
#utils.plot_utils.plotter3d(X,Y,Pol[0])

