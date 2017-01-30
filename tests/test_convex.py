from context import solvers, utils
from solvers import convex_envelope
from utils import plot_utils

import numpy as np

# Set up computational domain 
Nx = 21                      #grid size
dx = 1./(Nx-1)               #grid resolution
x = np.linspace(0.,1.,num = Nx) #gridpoints in x coordinate
X, Y = np.meshgrid(x,x,sparse=False,indexing='ij') # create x & y gridpoints, with matrix indexing

# Define an obstacle
def obstacle_fcn(X,Y):
    return -np.absolute(X*(X-.5)*(X-1))
# Calculate the obstacle and optionally plot it
G = obstacle_fcn(X,Y)
#plot_utils.plotter3d(X,Y,G)

U = convex_envelope.euler_step(G,dx)
plot_utils.plotter3d(X,Y,U)
