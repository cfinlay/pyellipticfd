from context import solvers, utils
from solvers import convex_envelope
from utils import plot_utils

import numpy as np

# Set up computational domain 
Nx = 41                      #grid size
dx = 2./(Nx-1)               #grid resolution
x = np.linspace(-1.,1.,num = Nx) #gridpoints in x coordinate
X, Y = np.meshgrid(x,x,sparse=False,indexing='ij') # create x & y gridpoints, with matrix indexing

# Define an obstacle
a = (-1/3,-2/3)
b = (1/3,2/3)
def obstacle_fcn(X,Y):
    return np.minimum(np.sqrt((X-a[0])**2 + (Y-a[1])**2),
                      np.sqrt((X-b[0])**2 + (Y-b[1])**2))

# Calculate the obstacle and optionally plot it
G = obstacle_fcn(X,Y)
#plot_utils.plotter3d(X,Y,G)

U = convex_envelope.euler_step(G,dx,max_iters=1e4)
plot_utils.plotter3d(X,Y,U)

#Utrue = np.zeros(U.size)
#Cases = (np.logical_or(X<=.25,X>=.75),
#         np.logical_and(X>.25,X<.75))
#
#Utrue[Cases[0]] = G[Cases[0]]
#Utrue[Cases[1]] = 0

